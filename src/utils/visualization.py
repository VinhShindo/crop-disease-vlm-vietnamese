import matplotlib.pyplot as plt


def plot_learning_curves(

    train_losses,
    val_losses,

    train_f1,
    val_f1,

    save_path
):

    epochs = range(
        1,
        len(train_losses) + 1
    )

    plt.figure(
        figsize=(12, 5)
    )

    # ------------------
    # Loss
    # ------------------

    plt.subplot(1, 2, 1)

    plt.plot(
        epochs,
        train_losses,
        label="Train"
    )

    plt.plot(
        epochs,
        val_losses,
        label="Validation"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.title(
        "Loss Curve"
    )

    plt.legend()

    # ------------------
    # F1
    # ------------------

    plt.subplot(1, 2, 2)

    plt.plot(
        epochs,
        train_f1,
        label="Train"
    )

    plt.plot(
        epochs,
        val_f1,
        label="Validation"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Macro F1")

    plt.title(
        "F1 Curve"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=300
    )

    plt.close()