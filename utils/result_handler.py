from pathlib import Path
from io import StringIO
from contextlib import redirect_stdout

from utils.evaluation import evaluate_model
import pickle



def save_evaluation(
    y_true,
    y_pred,
    y_scores,
    output_path,
    title='DT-IB ADAPTIVE MODEL'
):

    buffer = StringIO()

    with redirect_stdout(buffer):

        evaluate_model(
            y_true,
            y_pred,
            y_scores
        )

    evaluation_text = buffer.getvalue()

    print(evaluation_text)

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

        f.write(
            f"===== {title} =====\n\n"
        )

        f.write(evaluation_text)
        
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
        index=False
    )

    return output_path