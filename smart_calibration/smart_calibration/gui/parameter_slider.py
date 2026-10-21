"""Slider with a value label, bound to a float parameter of the calibration node."""

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QHBoxLayout, QLabel, QSlider, QWidget

# Slider moves within this time are sent to the calibration node as one parameter update.
SEND_DELAY_MS = 100


class ParameterSlider(QWidget):
    """
    Integer slider for one float parameter.

    to_position / from_position convert between parameter values and slider positions. User
    moves are passed to send(name, value) once the slider has rested for SEND_DELAY_MS.
    """

    def __init__(self, name, minimum, maximum, to_position, from_position, send,
                 value_format='{:.2f}', parent=None):
        super().__init__(parent)
        self.name = name
        self._to_position = to_position
        self._from_position = from_position
        self._send = send
        self._format = value_format

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(minimum, maximum)
        self.slider.setEnabled(False)  # until the calibration node's value is known
        self.slider.valueChanged.connect(self._on_moved)
        self.label = QLabel('-')
        self.label.setMinimumWidth(36)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.label)

        self._send_timer = QTimer(self)
        self._send_timer.setSingleShot(True)
        self._send_timer.setInterval(SEND_DELAY_MS)
        self._send_timer.timeout.connect(lambda: self._send(self.name, self.value()))

    def value(self):
        return self._from_position(self.slider.value())

    def busy(self):
        """Return True while the user is changing the value."""
        return self.slider.isSliderDown() or self._send_timer.isActive()

    def show_remote(self, value):
        """Show the calibration node's value (None: not known, disables the slider)."""
        if value is None:
            self.slider.setEnabled(False)
            self.label.setText('-')
            return
        self.slider.setEnabled(True)
        position = self._to_position(value)
        if position != self.slider.value():
            self.slider.blockSignals(True)  # not a user change: do not send it back
            self.slider.setValue(position)
            self.slider.blockSignals(False)
        self.label.setText(self._format.format(value))

    def _on_moved(self, position):
        self.label.setText(self._format.format(self._from_position(position)))
        self._send_timer.start()  # restarts while the slider keeps moving
