from pathlib import Path
from io import StringIO
from contextlib import redirect_stdout

from utils.evaluation import evaluate_model

import pickle


def save_evaluation(
    evaluations,
    output_path
):

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output_path,
        'w',
        encoding='utf-8'
    ) as f:

        for eval_item in evaluations:

            title = eval_item['title']

            y_true = eval_item['y_true']

            y_pred = eval_item['y_pred']

            y_scores = eval_item['y_scores']

            buffer = StringIO()

            with redirect_stdout(buffer):

                evaluate_model(
                    y_true,
                    y_pred,
                    y_scores
                )

            evaluation_text = (
                buffer.getvalue()
            )

            print(
                f"\n===== {title} =====\n"
            )

            print(evaluation_text)

            f.write(
                f"\n===== {title} =====\n\n"
            )

            f.write(evaluation_text)

            if (
                'statistical_testing'
                in eval_item
            ):

                stats = eval_item[
                    'statistical_testing'
                ]

                f.write(
                    "\n===== "
                    "STATISTICAL TESTING "
                    "=====\n\n"
                )

                f.write(
                    f"Wilcoxon Statistic : "
                    f"{stats['wilcoxon']:.4f}\n"
                )

                f.write(
                    f"P-Value            : "
                    f"{stats['p_value']:.6f}\n"
                )

                f.write(
                    f"Cohen's d          : "
                    f"{stats['cohens_d']:.4f}\n"
                )

                f.write(
                    f"Bonferroni Correct : "
                    f"{stats['bonferroni']:.6f}\n"
                )


def save_outputs(
    model_bundle,
    final_df,
    model_dir,
    result_dir
):

    with open(
        model_dir / 'tf_model_bundle.pkl',
        'wb'
    ) as f:

        pickle.dump(
            model_bundle,
            f
        )

    output_path = (
        result_dir
        / 'prediction_results.csv'
    )

    final_df.to_csv(
        output_path,
        sep=';',
        index=False
    )

    return output_path