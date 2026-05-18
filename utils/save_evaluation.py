from pathlib import Path
from io import StringIO
from contextlib import redirect_stdout

from utils.evaluation import evaluate_model


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
        
        