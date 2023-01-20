from abc import ABC


class Annotation(ABC):
    def __init__(self, text: str = None, label: str = None, start: int = None, end: int = None) -> None:
        """
        This class holds Tilt-annotations. The character span gives the position in the text. The text holds
        the concrete content of the annotation. The label gives the respective position in the tilt document.
        Currently the mapping has no direct mapping to the tilt document.

        Args:
            text (str): [description]
            label (str): [description]
            start (int): [description]
            end (int): [description]
        """
        self.annotated_text = text
        self.label = label
        self.start = start
        self.end = end


class SequenceAnnotation(Annotation):
    """Annotations on Sequence Level

    Args:
        Annotation ([type]): [description]
    """

    def __init__(self, text: str, label: str, start: int, end: int) -> None:
        super().__init__(text, label, start, end)


class TokenAnnotation(Annotation):
    """
    Annotations on Token Level

    Args:
        Annotation ([type]): [description]
    """

    def __init__(self, text: str, label: str, start: int, end: int) -> None:
        super().__init__(text, label, start, end)


class PredictedAnnotation(Annotation):

    """Annotation used for model predictions.
    """

    def __init__(self, text: str, label: str, start: int, end: int) -> None:
        super().__init__(text, label, start, end)

    def to_dict(self):
        return {
            "text": self.text,
            "label": self.label,
            "start": self.start,
            "end": self.end
        }
