import os

import fire
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from utils import ensure_number, load_config, setup_logging


def psnr(mse: Tensor, max_val=1.0) -> Tensor:
    """Computes PSNR based on input MSE value.

    Args:
        mse:
            The input MSE value. It must be <= max_val**2.
        max_val:
            Max possible value of tensor e.g. for uint8 images it's 255.
            Defaults to 1.0 for tensors normalized to (0.0,1.0).

    Returns:
        The computed PSNR value.
    """
    mse = ensure_number(
        num=mse,
        min_val=0.0,
        min_inclusive=True,
        max_val=max_val**2,
        max_inclusive=True,
    )
    if mse == 0.0:
        return float("inf")

    return 20 * torch.log10(torch.tensor(max_val)) - 10 * torch.log10(mse)


def init_siren(
    layer: nn.Linear,
    is_first: bool,
    w0: float,
    c_val: float = 6.0,
) -> None:
    """Applies SIREN initialization to a linear layer.

    For more info, see section 3.2 and supplemental materials of the paper.

    Args:
        layer:
            The input layer to be initialized.
        is_first:
            True if layer is the first layer of MLP otherwise False.
        w0:
            w0 in sin(w0*features).
        c_val:
            The variable "c" in the section 3.2 of the paper. Defaults to 6.0.
    """
    w0 = ensure_number(num=w0, min_val=0.0, min_inclusive=False)
    c_val = ensure_number(num=c_val, min_val=1.0, min_inclusive=True)

    if not hasattr(layer, "weight"):
        return
    with torch.no_grad():
        in_features = layer.weight.size(1)

        if is_first:
            bound = 1.0 / in_features
        else:
            bound = np.sqrt(c_val / in_features) / w0

        layer.weight.uniform_(-bound, bound)

        if layer.bias is not None:
            layer.bias.uniform_(-bound, bound)


class Sine(nn.Module):
    """The Sin activation module.

    Applies sin activation to input features.
    """

    def __init__(self, w0: float = 1.0):
        """Initializes the class.

        Args:
            w0:
                w0 in sin(w0*features).
        """
        super().__init__()
        self._w0 = ensure_number(num=w0, min_val=0.0, min_inclusive=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Runs forward pass.

        Args:
            x: Shape (B,N) where N is number of input features per batch.
                The tensor input to the sin acitivation.

        Returns:
            The output of sin activation.
        """
        return torch.sin(self._w0 * x)


class SIREN(nn.Module):
    """The core MLP model with sin activation."""

    def __init__(
        self,
        in_features: int,
        out_features: list[int],
        w0: float = 1.0,
        w0_initial: float = 30.0,
        bias: bool = True,
        c_val: float = 6.0,
        final_activation: nn.Module = None,  # e.g. nn.Sigmoid() for images
    ):
        """Builds the required layers followed by initialization.

        Args:
            in_features:
                First layers' expected number of input features.
            out_features:
                All layers' expected number of output features.
            w0:
                w0 in sin(w0*features). Defaults to 1.0.
            w0_initial:
                w0 in sin(w0*features) for the first layer. Defaults to 30.0.
            bias:
                Whether to use bias in linear layers. Defaults to True.
            c_val:
                The variable "c" in the section 3.2 of the paper. Defaults to
                6.0.
            final_activation:
                The activation be applied to the output of the final layer.
                Defaults to None.
        """
        super().__init__()
        # Ensure inputs are valid.
        in_features = ensure_number(
            num=in_features,
            min_val=1,
            min_inclusive=True,
        )
        # We need to at have at least 2 layers based on the paper.
        ensure_number(num=len(out_features), min_val=2, min_inclusive=True)

        for i, num_features in enumerate(out_features):
            out_features[i] = ensure_number(
                num=num_features, min_val=1, min_inclusive=True
            )

        self._w0 = ensure_number(num=w0, min_val=0.0, min_inclusive=False)
        self._w0_initial = ensure_number(
            num=w0_initial,
            min_val=0.0,
            min_inclusive=False,
        )
        self._c_val = ensure_number(num=c_val, min_val=1.0, min_inclusive=True)

        layers = []

        # First layer.
        layers.extend(
            [
                nn.Linear(in_features, out_features[0], bias=bias),
                Sine(w0=w0_initial),
            ]
        )

        # Hidden layers.
        for i in range(1, len(out_features) - 1):
            layers.extend(
                [
                    nn.Linear(out_features[i - 1], out_features[i], bias=bias),
                    Sine(w0=w0),
                ]
            )

        # Final layer.
        layers.append(nn.Linear(out_features[-2], out_features[-1], bias=bias))

        # Add (optional) final layer's activation.
        if final_activation is not None:
            layers.append(final_activation)

        # Initialize layers.
        self._init(layers)

        self._layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Runs the forward pass.

        Args:
            x: Shape (B,in_features)
                The input tensor.
        """
        return self._layers(x)

    def _init(self, layers: list[nn.Module]) -> None:
        """Initializes the model's first and remaining layers.

        The initialization is done based on section 3.2 and supramental
        materials in the paper.

        Args:
            layers:
                The layers of the model.
        """
        for i, layer in enumerate(layers):
            if i == 0:
                init_siren(
                    layer=layer,
                    is_first=True,
                    w0=self._w0_initial,
                    c_val=self._c_val,
                )
            else:
                init_siren(
                    layer=layer,
                    is_first=False,
                    w0=self._w0,
                    c_val=self._c_val,
                )


def transform_image_to_pixel_location_colors(
    image: Image, size: tuple[int, int] | None
) -> dict[str, Tensor]:
    """Transforms image and returns pixel locations and colors.

    The function first optionally resize the image and the converts it to
    tensor. Then the location and colors of all pixels are returned as
    flattened tensor along with the corresponding tensor image.

    Args:
        image: Shape: (H,W,3)
            Input PIL image.
        size:
            The (height, width) that the image will be resized to.

    Returns:
        Dictionary containing:
            coords: Shape (N,2)
                The pixel locations.
            colors: Shape (N,3)
                The pixel colors.
            image: Shape (H,W,3)
                The resized image.
    """
    if size is not None:
        # Ensure size has only 2 elements and each is >= 1.
        ensure_number(
            num=len(size),
            min_val=2,
            max_val=2,
            min_inclusive=True,
            max_inclusive=2,
        )
        ensure_number(num=size[0], min_val=1, min_inclusive=True)
        ensure_number(num=size[1], min_val=1, min_inclusive=True)

    # Compose transforms to be applied to the input image.
    transform = transforms.Compose(
        []
        if size is None
        else [transforms.Resize(size)]
        + [
            transforms.ToTensor(),  # Normalize input to [0,1] range
        ]
    )

    # Apply transforms.
    img = transform(image)  # Output shape: (3, H, W)

    _, height, width = img.shape

    # (H*W, 3)
    colors = img.permute(1, 2, 0).reshape(-1, 3)

    # Make tensor containing all pixel locations.
    ys, xs = torch.meshgrid(
        torch.linspace(-1, 1, height),
        torch.linspace(-1, 1, width),
        indexing="ij",
    )
    coords = torch.stack([xs, ys], dim=-1).reshape(-1, 2)

    return {
        "coords": coords,
        "colors": colors,
        "image": img.permute(1, 2, 0),  # Convert back to (H,W,3) shape
    }


@torch.no_grad()
def reconstruct_image(
    model: nn.Module,
    coords: Tensor,
    height: int,
    width: int,
    device: torch.device,
) -> Tensor:
    """Runs a tensor of coordinates through model and reconstructs the image.

    Args:
        model:
            The model to be used to predict the colors.
        coords: Shape (N,2)
            The pixel coordinates the colors will be predicted for.
        height:
            The height of output image.
        width:
            The width of output image.
        device:
            The compute resource to be used.

    Returns:
        Shape (H,W,3)
        The reconstructed image.
    """
    model.eval()
    preds = model(coords.to(device))
    img = preds.cpu().reshape(height, width, 3).clamp(0, 1)
    return img


class SirenNetwork:
    """The module training a Siren model.

    The module first makes the following directories under out_dir in
    configuration file or default under out_dir={this package
    directory}/outputs:
        {out_dir}/log:
            Where the logging will be saved into.
        {out_dir}/summaries:
            Where the tensorboard summaries will be saved into.
        {out_dir}/model:
            Where the model checkpoints will be saved into.



    Attributes:
        config:
            The configurations.
        logger:
            The logging module.
        device:
            The compute resource.
        epoch:
            The number of epochs the model is trained over.
        model:
            The Siren model.
        optimizer:
            The optimizer being used.
    """

    def __init__(self, config_name: str):
        """Initializes the model.

        Args:
            config_name: The name of the configuration file.
        """
        # Load the configurations.
        if not config_name.endswith(".yaml"):
            config_name = config_name + ".yaml"
        self.config = load_config(config_name)

        # Make the required output directories.
        self._make_output_dirs()

        # Setup the logging.
        self.logger = setup_logging(
            path=self.config["log_path"],
            level=self.config["logging_level"],
        )

        # Find the available compute resource
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu",
        )

    def run(self) -> None:
        """Runs the training process."""

        # Load data
        image = Image.open(self.config["data_path"]).convert("RGB")

        data_info = transform_image_to_pixel_location_colors(
            image,
            size=tuple(self.config["input_size"]),
        )
        coords = data_info["coords"]
        colors = data_info["colors"]
        image = data_info["image"]

        # Build Dataset.
        dataset = TensorDataset(coords, colors)

        # Make DataLoader from the dataset. Set batch size to a very large
        # value while avoiding Out Of Memory Error.
        loader = DataLoader(
            dataset,
            batch_size=int(self.config["batch_size_ratio"] * coords.shape[0]),
            shuffle=True,
        )

        # Build model and use sigmoid activation for the last layer since we
        # are training over images.
        self.model = SIREN(
            final_activation=nn.Sigmoid(),  # important for images
            **self.config["model"],
        )

        # increment epoch if resuming the train
        self.epoch = 0

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            **self.config["optimizer"],
        )

        # Resume the network state based on the input checkpoint id.
        self._resume(self.config["resume_ckpt_id"])

        self.model = self.model.to(self.device)

        loss_function = nn.MSELoss()

        writer = SummaryWriter(log_dir=self.config["summaries"])

        while self.epoch < self.config["num_epochs"]:
            self.model.train()
            epoch_loss = 0

            for batch_coords, batch_pixels in loader:
                batch_coords = batch_coords.to(self.device)
                batch_pixels = batch_pixels.to(self.device)

                preds = self.model(batch_coords)
                loss = loss_function(preds, batch_pixels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            epoch_loss /= len(loader)

            if self.epoch > 0 and self.epoch % self.config["ckpt_period"] == 0:
                # Save network state
                self._save_state()
                self.logger.info("Saved checkpoint.")

            if (
                self.epoch > 0
                and self.epoch % self.config["summary_scalar_period"] == 0
            ):
                # Compute PSNR and write both loss and PSNR to tensorboard.
                writer.add_scalar("Scalars/Loss", epoch_loss, self.epoch)

                psnr_val = psnr(mse=torch.tensor(epoch_loss), max_val=1.0)
                self.logger.info(
                    f"Epoch: {self.epoch} | Loss: {epoch_loss:.6f} | PSNR: {psnr_val:.6f}",
                )

                writer.add_scalar("Scalars/PSNR", psnr_val, self.epoch)

            if self.epoch > 0 and self.epoch % self.config["summary_image_period"] == 0:
                # Predict colors and write resulting image to tensorboard.
                image_pred = reconstruct_image(
                    model=self.model,
                    coords=coords,
                    height=image.shape[0],
                    width=image.shape[1],
                    device=self.device,
                )
                writer.add_image(
                    tag="Data/pred",
                    img_tensor=image_pred,
                    global_step=self.epoch,
                    dataformats="HWC",
                )

            self.epoch += 1

        # Save last epoch.
        self._save_state()

        writer.close()

        self.logger.info("Successfully processed training.")

    def _make_output_dirs(self) -> None:
        """Makes the output file directories.

        See above class description for more details.
        """
        paths_to_make = [
            self.config["out_dir"],
            self.config["log_dir"],
            self.config["model_dir"],
        ]
        for path in paths_to_make:
            os.makedirs(path, exist_ok=True)

    def _make_ckpt_path(self, ckpt_id: int) -> str:
        """Generates the path using checkpoint id.

        Args:
            ckpt_id:
                The checkpoint id.

        Returns:
            The generated path.
        """
        path = os.path.join(
            self.config["model_dir"],
            "{:06d}.ckpt".format(ckpt_id),
        )
        return path

    def _save_state(self) -> None:
        """Save network state.

        The following items will be saved:
            epoch:
                The current epoch number.
            model:
                The Siren model information.
            optimizer:
                The optimizer information used for training.

        """
        path = self._make_ckpt_path(self.epoch)

        state = {
            "epoch": self.epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(state, path)

    def _resume(self, ckpt_id: int | None) -> None:
        """Loads the a checkpoint and resumes training.

        It loads the epoch id, the model and the optimizer info.

        Args:
            ckpt_id:
                The checkpoint id to be loaded.
        """
        if ckpt_id is None:
            self.logger.info("Resume checkpoint id is None.")
            return

        ckpt_path = self._make_ckpt_path(ckpt_id)
        ckpt = torch.load(ckpt_path)

        # Load the last epoch processed.
        self.epoch = ckpt["epoch"]

        # Load optimizer info.
        self.optimizer.load_state_dict(ckpt["optimizer"])

        # Load model.
        self.model.load_state_dict(ckpt["model"])

        self.logger.info(f"Resumed network state from {ckpt_path}")


def run_siren_network(config_name: str = "image.yaml") -> None:
    """Runs the full training process of the Siren model.

    The configs are saved under package's config folder.
    To run the training, run:

        $ python {path to train.py} --config_name={path/to/your/config.yaml}

    For example for run the following for camera man image (or modify it for
    your own image) for the directory of this file:

        $ python train.py --config_name=image.yaml

        or:

        $ python train.py --config_name=image

    Args:
        config_name: The name of the input configuration files.
    """
    siren = SirenNetwork(config_name)
    siren.run()


if __name__ == "__main__":
    fire.Fire(run_siren_network)
