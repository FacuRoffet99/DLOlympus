import pytest
import matplotlib.pyplot as plt

from src.DLOlympus.utils.plots import plot_confusion_matrix

def test_plot_confusion_matrix_with_integer_inputs():
    """Checks if the function runs correctly on integer-encoded inputs."""
    gts = [0, 1, 2, 0, 1, 2]
    preds = [0, 1, 2, 0, 1, 2]
    classes = ['cat', 'dog', 'bird']
    
    fig = plot_confusion_matrix(gts, preds, classes)
    
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

def test_plot_confusion_matrix_with_string_inputs():
    """Checks if the function runs correctly on string-decoded inputs."""
    classes = ['cat', 'dog', 'bird']
    gts = ['cat', 'dog', 'bird', 'cat', 'dog', 'bird']
    preds = ['cat', 'dog', 'bird', 'cat', 'dog', 'bird']
    
    fig = plot_confusion_matrix(gts, preds, classes)
    
    assert isinstance(fig, plt.Figure)
    assert fig.axes[0].get_xticklabels()[0].get_text() == 'cat'
    plt.close(fig)

def test_plot_confusion_matrix_saves_file_correctly(tmp_path):
    """Verifies that the plot is saved to the specified path."""
    gts = [0, 1]
    preds = [1, 1]
    classes = ['zero', 'one']
    save_path = tmp_path / "cm.png"
    
    fig = plot_confusion_matrix(gts, preds, classes, path=str(save_path))
    
    assert save_path.is_file()
    plt.close(fig)

def test_plot_confusion_matrix_raises_error_on_empty_input():
    """Ensures the function raises a ValueError for empty inputs."""
    with pytest.raises(ValueError):
        plot_confusion_matrix([], [], ['class_a'])