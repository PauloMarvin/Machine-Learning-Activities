import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import colorsys
from matplotlib.colors import ListedColormap

from utils.dataset_utils import DatasetUtils


class ImageGenerator:

    def __init__(self, figure_size: tuple[int, int], font_size: int):
        self.figure_size = figure_size
        self.font_size = font_size

    def create_boxplot(
            self,
            vector_distribution: pd.Series | list[float],
            y_label: str,
            image_title: str,
            path_to_save: str,
            show: bool = True,
    ) -> None:
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=self.figure_size)
        sns.boxplot(
            y=vector_distribution,
            width=0.4,
            color="skyblue",
            linewidth=2,
            fliersize=5,
        )

        plt.title(label=image_title, fontsize=self.font_size)
        plt.ylabel(ylabel=y_label, fontsize=self.font_size)
        plt.ylim(min(vector_distribution) - 0.1, 1)
        plt.yticks(fontsize=self.font_size)

        plt.savefig(path_to_save, format="jpeg")

        if show:
            plt.show()

        plt.close()

    def create_confusion_matrix_heatmap(
            self,
            confusion_matrix: np.ndarray,
            x_label: str,
            y_label: str,
            image_title: str,
            path_to_save: str,
            show: bool = True,
    ) -> None:
        plt.figure(figsize=self.figure_size)
        sns.heatmap(
            confusion_matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            annot_kws={"size": self.font_size},
        )

        plt.xlabel(xlabel=x_label, fontsize=self.font_size)
        plt.ylabel(ylabel=y_label, fontsize=self.font_size)
        plt.title(label=image_title, fontsize=self.font_size)
        plt.xticks(fontsize=self.font_size)
        plt.yticks(fontsize=self.font_size)

        plt.savefig(path_to_save, format="jpeg")

        if show:
            plt.show()

        plt.close()

    def create_decision_surface(
            self,
            X_train: pd.DataFrame,
            y_train: pd.Series | np.ndarray,
            y_pred: np.ndarray,
            image_title: str,
            feature_x: str,
            feature_y: str,
            resolution_points: int,
            path_to_save: str,
            num_classes: int,
            show: bool = True,
    ) -> None:
        xx, yy = DatasetUtils.create_meshgrid(
            X_train,
            feature_x,
            feature_y,
            resolution_points=resolution_points,
        )

        feature_x_max, feature_x_min, feature_y_max, feature_y_min = (
            DatasetUtils.calculate_feature_bounds(
                X_train,
                feature_x,
                feature_y,
            )
        )

        cmap_light = ListedColormap(self.generate_distinct_pastel_colors(num_classes))
        plt.figure(figsize=self.figure_size)
        plt.pcolormesh(xx, yy, y_pred, cmap=cmap_light)
        sc = plt.scatter(
            X_train[feature_x],
            X_train[feature_y],
            c=y_train,
            edgecolor="k",
            s=50,
            cmap=cmap_light,
        )
        plt.legend(*sc.legend_elements(), title="Classes", loc="upper right")

        plt.xlabel(feature_x, fontsize=self.font_size)
        plt.ylabel(feature_y, fontsize=self.font_size)
        plt.title(image_title, fontsize=self.font_size)

        plt.xlim(feature_x_min, feature_x_max)
        plt.ylim(feature_y_min, feature_y_max)

        plt.xticks(fontsize=self.font_size)
        plt.yticks(fontsize=self.font_size)

        plt.savefig(path_to_save, format="jpeg")

        if show:
            plt.show()

        plt.close()

    @staticmethod
    def generate_distinct_pastel_colors(number_of_colors: int) -> list[str]:
        colors = []
        for i in range(number_of_colors):
            hue = i / number_of_colors
            r, g, b = colorsys.hls_to_rgb(hue, 0.8, 0.5)  # Mais luminosidade e menos saturação
            colors.append('#{:02X}{:02X}{:02X}'.format(int(r * 255), int(g * 255), int(b * 255)))
        return colors

