import torch
import numpy as np
import tqdm
from roma.datasets.carblender import CarBlenderBuilder
from roma.utils import warp_kpts
from torch.utils.data import ConcatDataset, WeightedRandomSampler
import roma
from PIL import Image

import mlflow
mlflow.autolog()

def concatenate_images_horizontally(*images):
    # Get the width and height of each image
    widths, heights = zip(*(img.size for img in images))

    # Calculate the total width and the maximum height
    total_width = sum(widths)
    max_height = max(heights)

    # Create a new blank image with the calculated dimensions
    new_image = Image.new('RGB', (total_width, max_height))

    # Paste each image into the new image
    x_offset = 0
    for img in images:
        new_image.paste(img, (x_offset, 0))
        x_offset += img.width

    return new_image

class CarBlenderBenchmark:
    def __init__(self, data_root, h=728, w=966, num_samples = 1000) -> None:
        carblender = CarBlenderBuilder(data_root, ht=h, wt=w)
        self.dataset = ConcatDataset(
            carblender.build(start=100) ## last 8 scenes
        )  # fixed resolution of 384,512
        self.num_samples = num_samples

    def geometric_dist(self, depth1, depth2, T_1to2, K1, K2, dense_matches):
        b, h1, w1, d = dense_matches.shape
        with torch.no_grad():
            x1 = dense_matches[..., :2].reshape(b, h1 * w1, 2)
            mask, x2 = warp_kpts(
                x1.double(),
                depth1.double(),
                depth2.double(),
                T_1to2.double(),
                K1.double(),
                K2.double(),
            )
            x2 = torch.stack(
                (w1 * (x2[..., 0] + 1) / 2, h1 * (x2[..., 1] + 1) / 2), dim=-1
            )
            prob = mask.float().reshape(b, h1, w1)
        x2_hat = dense_matches[..., 2:]
        x2_hat = torch.stack(
            (w1 * (x2_hat[..., 0] + 1) / 2, h1 * (x2_hat[..., 1] + 1) / 2), dim=-1
        )
        gd = (x2_hat - x2.reshape(b, h1, w1, 2)).norm(dim=-1)
        gd = gd[prob == 1]
        pck_1 = (gd < 1.0).float().mean()
        pck_3 = (gd < 3.0).float().mean()
        pck_5 = (gd < 5.0).float().mean()
        return gd, pck_1, pck_3, pck_5, prob

    def benchmark(self, model, batch_size=8):
        model.train(False)
        with torch.no_grad():
            gd_tot = 0.0
            pck_1_tot = 0.0
            pck_3_tot = 0.0
            pck_5_tot = 0.0
            sampler = WeightedRandomSampler(
                torch.ones(len(self.dataset)), replacement=False, num_samples=self.num_samples
            )
            B = batch_size
            dataloader = torch.utils.data.DataLoader(
                self.dataset, batch_size=B, num_workers=batch_size, sampler=sampler
            )
            for idx, data in tqdm.tqdm(enumerate(dataloader), disable = roma.RANK > 0):
                im_A, im_B, depth1, depth2, T_1to2, K1, K2 = (
                    data["im_A"],
                    data["im_B"],
                    data["im_A_depth"].cuda(),
                    data["im_B_depth"].cuda(),
                    data["T_1to2"].cuda(),
                    data["K1"].cuda(),
                    data["K2"].cuda(),
                )
                matches, certainty = model.match(im_A, im_B, batched=True)
                gd, pck_1, pck_3, pck_5, prob = self.geometric_dist(
                    depth1, depth2, T_1to2, K1, K2, matches
                )
                
                from roma.utils.utils import tensor_to_pil
                import torch.nn.functional as F
                path = "vis"
                H, W = model.get_output_resolution()
                white_im = torch.ones((B,1,H,W),device="cuda")
                im_B_transfer_rgb = F.grid_sample(
                    im_B.cuda(), matches[:,:,:W, 2:], mode="bilinear", align_corners=False
                )
                warp_im = im_B_transfer_rgb
                c_b = certainty[:,None]#(certainty*0.9 + 0.1*torch.ones_like(certainty))[:,None]
                vis_im = c_b * warp_im + (1 - c_b) * white_im
                if idx % 50 == 0:
                    for b in range(B):
                        vis_warp = tensor_to_pil(vis_im[b], unnormalize=True)
                        im_a = tensor_to_pil(im_A[b].cuda(), unnormalize=True)
                        im_b = tensor_to_pil(im_B[b].cuda(), unnormalize=True)
                        # concatenate images
                        all_images = concatenate_images_horizontally(im_a, im_b, vis_warp)
                        mlflow.log_image(all_images, f"image_{idx}_{b}.png")
                        
                gd_tot, pck_1_tot, pck_3_tot, pck_5_tot = (
                    gd_tot + gd.mean(),
                    pck_1_tot + pck_1,
                    pck_3_tot + pck_3,
                    pck_5_tot + pck_5,
                )
        return {
            "epe": gd_tot.item() / len(dataloader),
            "carblender_pck_1": pck_1_tot.item() / len(dataloader),
            "carblender_pck_3": pck_3_tot.item() / len(dataloader),
            "carblender_pck_5": pck_5_tot.item() / len(dataloader),
        }
