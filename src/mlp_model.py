"""Model 2: PyTorch MLP (deep learning).

Dataset class, training loop with BCEWithLogitsLoss and AdamW, early stopping
on inner validation split PR AUC, seed control, random search with tuned
class weighting, final refit at the median best epoch, and final models
trained on the seeds in config.yaml.

Like baselines.py, this module holds no preprocessing of its own. Every
function that fits takes a preprocessor: a callable with no arguments that
returns an unfitted sklearn transformer, which takes and returns a DataFrame.
In each fold a new one is fitted on the fitting rows only. The MLP then
standardises the preprocessed columns with a StandardScaler fitted on the
same rows (plan Section 4, preprocessing step 6).

Inside cross validation, inner_split from split.py holds out 15% of the
fold's fitting rows for early stopping. The preprocessor and scaler are
fitted on the remaining 85%, so the early stopping rows are scored like
unseen data and the fold being scored is never touched.

Settings for the search live under mlp in config.yaml. Nothing is saved
here; save_oof_predictions in evaluate.py does that.

Owner: Zack.
"""

import logging
import random
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.base import TransformerMixin
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.baselines import oof_scores
from src.config import load_config
from src.evaluate import cv_metrics
from src.load import LABEL
from src.split import inner_split

logger = logging.getLogger(__name__)

Preprocessor = Callable[[], TransformerMixin]


@dataclass(frozen=True)
class MLPParams:
    """One configuration of the MLP, as sampled by the random search.

    Attributes:
        class_weighting: If True, pos_weight is negatives / positives in the
            rows the network is trained on; if False, pos_weight is 1.
        n_layers: Number of hidden layers, 1 to 3.
        width: Units in every hidden layer.
        dropout: Dropout probability after every hidden layer.
        learning_rate: AdamW learning rate.
        weight_decay: AdamW weight decay.
    """

    class_weighting: bool
    n_layers: int
    width: int
    dropout: float
    learning_rate: float
    weight_decay: float


class RatioDataset(Dataset):
    """Preprocessed ratios and labels as float32 tensors for a DataLoader."""

    def __init__(self, features: np.ndarray, labels: np.ndarray) -> None:
        """Store the rows as tensors.

        Args:
            features: One row per statement, already preprocessed and scaled.
            labels: Class label of each row, 1 = bankrupt.
        """
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        """Return the number of rows."""
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the features and label of one row."""
        return self.features[index], self.labels[index]


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch so a run can be repeated exactly.

    Args:
        seed: The seed, taken from config.yaml by the caller.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_network(n_inputs: int, params: MLPParams) -> nn.Sequential:
    """Build the MLP: hidden layers of Linear, ReLU and Dropout, then one logit.

    Args:
        n_inputs: Number of columns after preprocessing.
        params: The configuration, for the depth, width and dropout.

    Returns:
        An untrained network that outputs one logit per row.
    """
    layers: list[nn.Module] = []
    size = n_inputs
    for _ in range(params.n_layers):
        layers += [nn.Linear(size, params.width), nn.ReLU(), nn.Dropout(params.dropout)]
        size = params.width
    layers.append(nn.Linear(size, 1))
    return nn.Sequential(*layers)


def pos_weight(labels: np.ndarray, params: MLPParams) -> torch.Tensor:
    """Compute the positive class weight for BCEWithLogitsLoss.

    Args:
        labels: Labels of the rows the network is trained on.
        params: The configuration, for whether class weighting is on.

    Returns:
        negatives / positives in labels if class weighting is on, else 1, as
        a one element tensor.
    """
    if not params.class_weighting:
        return torch.tensor([1.0])
    positives = labels.sum()
    return torch.tensor([(len(labels) - positives) / positives], dtype=torch.float32)


def predict_logits(network: nn.Module, features: np.ndarray) -> np.ndarray:
    """Score rows with a network in evaluation mode (dropout off).

    Args:
        network: A trained network from build_network.
        features: Preprocessed and scaled rows.

    Returns:
        One logit per row; higher means more likely bankrupt.
    """
    network.eval()
    with torch.no_grad():
        inputs = torch.as_tensor(features, dtype=torch.float32)
        return network(inputs).squeeze(1).numpy()


def train_network(
    train_x: np.ndarray,
    train_y: np.ndarray,
    params: MLPParams,
    seed: int,
    config: dict[str, Any],
    stop_x: np.ndarray | None = None,
    stop_y: np.ndarray | None = None,
    n_epochs: int | None = None,
) -> tuple[nn.Sequential, int]:
    """Train one network, with early stopping or for a fixed number of epochs.

    With stop_x and stop_y, the network trains for up to mlp.max_epochs and
    stops once PR AUC on the early stopping rows has not improved for
    mlp.patience epochs; the weights of the best epoch are restored. Without
    them, it trains for exactly n_epochs, as in the final refit.

    Args:
        train_x: Preprocessed and scaled rows to train on.
        train_y: Their labels.
        params: The configuration.
        seed: Seed for the weight initialisation, dropout and batch order.
        config: Parsed project configuration, for the mlp settings.
        stop_x: Preprocessed and scaled early stopping rows, or None.
        stop_y: Their labels, or None.
        n_epochs: Epochs to train for when there are no early stopping rows.

    Returns:
        The trained network and its best epoch, counted from 1 (n_epochs
        when training for a fixed number of epochs).

    Raises:
        ValueError: If neither early stopping rows nor n_epochs are given.
    """
    early_stopping = stop_x is not None and stop_y is not None
    if not early_stopping and n_epochs is None:
        raise ValueError("give early stopping rows or a fixed n_epochs")
    settings = config["mlp"]
    torch.set_num_threads(config["num_threads"])
    set_seed(seed)

    network = build_network(train_x.shape[1], params)
    loss_function = nn.BCEWithLogitsLoss(pos_weight=pos_weight(train_y, params))
    optimiser = torch.optim.AdamW(
        network.parameters(), lr=params.learning_rate, weight_decay=params.weight_decay
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        RatioDataset(train_x, train_y),
        batch_size=settings["batch_size"],
        shuffle=True,
        generator=generator,
    )

    max_epochs = settings["max_epochs"] if early_stopping else n_epochs
    best_pr_auc = -np.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] = {}
    for epoch in range(1, max_epochs + 1):
        network.train()
        for features, labels in loader:
            optimiser.zero_grad()
            loss = loss_function(network(features).squeeze(1), labels)
            loss.backward()
            optimiser.step()

        if not early_stopping:
            continue
        pr_auc = average_precision_score(stop_y, predict_logits(network, stop_x))
        if pr_auc > best_pr_auc:
            best_pr_auc, best_epoch = pr_auc, epoch
            best_state = {k: v.clone() for k, v in network.state_dict().items()}
        elif epoch - best_epoch >= settings["patience"]:
            break

    if not early_stopping:
        return network, n_epochs
    network.load_state_dict(best_state)
    return network, best_epoch


def fit_transformers(
    preprocessor: Preprocessor, x: pd.DataFrame, y: pd.Series
) -> tuple[TransformerMixin, StandardScaler]:
    """Fit a new preprocessor and a StandardScaler on the given rows only.

    Args:
        preprocessor: Returns a new unfitted preprocessing transformer.
        x: Ratios of the rows to fit on.
        y: Their labels.

    Returns:
        The fitted preprocessor and scaler.
    """
    transformer = preprocessor().fit(x, y)
    scaler = StandardScaler().fit(transformer.transform(x))
    return transformer, scaler


def transform(
    transformer: TransformerMixin, scaler: StandardScaler, x: pd.DataFrame
) -> np.ndarray:
    """Apply a fitted preprocessor and scaler.

    Args:
        transformer: Fitted preprocessor from fit_transformers.
        scaler: Fitted scaler from fit_transformers.
        x: Ratios to transform.

    Returns:
        The preprocessed, standardised rows as a float32 array.
    """
    return scaler.transform(transformer.transform(x)).astype(np.float32)


def oof_mlp(
    stem: str,
    preprocessor: Preprocessor,
    params: MLPParams,
    config: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, list[int]]:
    """Make the out of fold predictions of one MLP configuration.

    In each of the 15 folds from cv_folds, inner_split takes the early
    stopping rows from the fitting rows, the preprocessor and scaler are
    fitted on the rest, and the network trains with early stopping on the
    nominated seed. The scored fold is then scored with the best epoch.

    Args:
        stem: File name without extension, e.g. 5year.
        preprocessor: Returns a new unfitted preprocessing transformer.
        params: The configuration.
        config: Parsed project configuration. Read from config.yaml if None.

    Returns:
        The out of fold predictions in PREDICTION_COLUMNS, scorer mlp, with
        the logit as the score; and the best epoch of each fold, in fold
        order.
    """
    if config is None:
        config = load_config()
    seed = config["seeds"]["nominated"]
    best_epochs: list[int] = []

    def fit_and_score(
        fit_x: pd.DataFrame, fit_y: pd.Series, score_x: pd.DataFrame
    ) -> np.ndarray:
        train_ids, stop_ids = inner_split(fit_x.assign(**{LABEL: fit_y}), config)
        transformer, scaler = fit_transformers(
            preprocessor, fit_x.loc[train_ids], fit_y.loc[train_ids]
        )
        network, best_epoch = train_network(
            transform(transformer, scaler, fit_x.loc[train_ids]),
            fit_y.loc[train_ids].to_numpy(),
            params,
            seed,
            config,
            stop_x=transform(transformer, scaler, fit_x.loc[stop_ids]),
            stop_y=fit_y.loc[stop_ids].to_numpy(),
        )
        best_epochs.append(best_epoch)
        return predict_logits(network, transform(transformer, scaler, score_x))

    predictions = oof_scores(stem, "mlp", fit_and_score, config)
    return predictions, best_epochs


def sample_configurations(config: dict[str, Any]) -> list[MLPParams]:
    """Draw the random search configurations from the space in config.yaml.

    Each setting is drawn independently with the nominated seed, so the same
    configurations come out on every run. Weight decay is drawn log
    uniformly between its two bounds.

    Args:
        config: Parsed project configuration, for mlp.search,
            mlp.n_configurations and the nominated seed.

    Returns:
        mlp.n_configurations configurations, in the order they were drawn.
    """
    space = config["mlp"]["search"]
    rng = np.random.default_rng(config["seeds"]["nominated"])
    low, high = np.log10(space["weight_decay"])
    configurations = []
    for _ in range(config["mlp"]["n_configurations"]):
        configurations.append(
            MLPParams(
                class_weighting=bool(rng.choice(space["class_weighting"])),
                n_layers=int(rng.choice(space["n_layers"])),
                width=int(rng.choice(space["width"])),
                dropout=float(rng.choice(space["dropout"])),
                learning_rate=float(rng.choice(space["learning_rate"])),
                weight_decay=float(10 ** rng.uniform(low, high)),
            )
        )
    return configurations


def search_mlp(
    stem: str,
    preprocessor: Preprocessor,
    config: dict[str, Any] | None = None,
    configurations: list[MLPParams] | None = None,
) -> tuple[MLPParams, pd.DataFrame, pd.DataFrame, int]:
    """Run the random search and choose the configuration with the best CV PR AUC.

    Every configuration is run on all 15 folds with oof_mlp. The one with the
    highest mean PR AUC over the folds (from cv_metrics) is chosen; on a tie
    the one drawn first is kept.

    Args:
        stem: File name without extension, e.g. 5year.
        preprocessor: Returns a new unfitted preprocessing transformer.
        config: Parsed project configuration. Read from config.yaml if None.
        configurations: Configurations to try. Drawn with
            sample_configurations if None.

    Returns:
        The chosen configuration; one row per configuration with its settings,
        mean and standard deviation of PR AUC and median best epoch; the out
        of fold predictions of the chosen configuration; and its median best
        epoch, the epoch count for the final refit.
    """
    if config is None:
        config = load_config()
    if configurations is None:
        configurations = sample_configurations(config)

    rows = []
    predictions = {}
    for number, params in enumerate(configurations):
        predictions[number], best_epochs = oof_mlp(stem, preprocessor, params, config)
        metrics = cv_metrics(predictions[number])
        rows.append(
            {
                "configuration": number,
                **asdict(params),
                "mean_pr_auc": metrics["mean"]["pr_auc"],
                "std_pr_auc": metrics["std"]["pr_auc"],
                "median_best_epoch": median_epoch(best_epochs),
            }
        )
        logger.info(
            "%s mlp configuration %d: mean PR AUC %.4f",
            stem,
            number,
            metrics["mean"]["pr_auc"],
        )

    results = pd.DataFrame(rows)
    # idxmax returns the first highest value, so a tie keeps the earlier draw.
    best = int(results["mean_pr_auc"].idxmax())
    logger.info("%s mlp: chose configuration %d", stem, best)
    return (
        configurations[best],
        results,
        predictions[best],
        int(results.loc[best, "median_best_epoch"]),
    )


def median_epoch(best_epochs: list[int]) -> int:
    """Return the median best epoch, rounded to a whole epoch.

    Args:
        best_epochs: Best epoch of each fold.

    Returns:
        The median, rounded half to even by Python's round, and at least 1.
    """
    return max(1, round(float(np.median(best_epochs))))


@dataclass
class FittedMLP:
    """A final MLP with the preprocessor and scaler it was trained behind."""

    transformer: TransformerMixin
    scaler: StandardScaler
    network: nn.Sequential

    def score(self, x: pd.DataFrame) -> np.ndarray:
        """Return the logit risk score of each row of raw ratios.

        Args:
            x: Raw ratios, Attr1 to Attr64.

        Returns:
            One logit per row; higher means more likely bankrupt.
        """
        return predict_logits(self.network, transform(self.transformer, self.scaler, x))


def fit_final_models(
    training: pd.DataFrame,
    preprocessor: Preprocessor,
    params: MLPParams,
    n_epochs: int,
    features: list[str],
    config: dict[str, Any] | None = None,
) -> dict[int, FittedMLP]:
    """Refit the chosen configuration on all training rows, once per seed.

    The preprocessor and scaler are fitted once on all training rows; each
    network then trains for exactly n_epochs (the median best epoch from CV)
    on a seed from seeds.models. The seed 42 model drives every headline
    result; the others are stability evidence only.

    Args:
        training: All training rows of one file, from load_training_set.
        preprocessor: Returns a new unfitted preprocessing transformer.
        params: The chosen configuration.
        n_epochs: Epochs to train each network for.
        features: Ratio columns to use, normally FEATURES from load.py.
        config: Parsed project configuration. Read from config.yaml if None.

    Returns:
        One FittedMLP per seed in seeds.models, keyed by seed.
    """
    if config is None:
        config = load_config()
    x, y = training[features], training[LABEL]
    transformer, scaler = fit_transformers(preprocessor, x, y)
    train_x = transform(transformer, scaler, x)
    models = {}
    for seed in config["seeds"]["models"]:
        network, _ = train_network(
            train_x, y.to_numpy(), params, seed, config, n_epochs=n_epochs
        )
        models[seed] = FittedMLP(transformer, scaler, network)
    return models
