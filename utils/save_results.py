import pickle


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