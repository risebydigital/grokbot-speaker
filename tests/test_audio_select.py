"""Speaker choice: USB playback, then the 3.5 mm jack. Not the microphone, not Bluetooth."""

from gva.runtime.audio import prefer_output


def _dev(index, name, inputs=0, outputs=2):
    return {"index": index, "name": name, "inputs": inputs, "outputs": outputs}


def test_usb_playback_beats_the_headphone_jack():
    devices = [
        _dev(0, "AIRHUG", inputs=2, outputs=0),
        _dev(1, "bcm2835 Headphones", outputs=2),
        _dev(2, "USB Audio", outputs=2),
    ]
    assert prefer_output(devices, "") == 2


def test_headphone_jack_when_no_usb_speaker():
    devices = [
        _dev(0, "AIRHUG", inputs=2, outputs=0),
        _dev(1, "bcm2835 Headphones", outputs=2),
    ]
    assert prefer_output(devices, "") == 1


def test_configured_substring_wins():
    devices = [
        _dev(1, "bcm2835 Headphones", outputs=2),
        _dev(2, "USB Audio", outputs=2),
    ]
    assert prefer_output(devices, "headphone") == 1


def test_airhug_and_bluetooth_are_not_speakers():
    devices = [
        _dev(0, "AIRHUG", inputs=2, outputs=2),
        _dev(3, "bluez_output.Bluetooth", outputs=2),
        _dev(1, "bcm2835 Headphones", outputs=2),
    ]
    assert prefer_output(devices, "") == 1
    assert prefer_output(devices, "bluetooth") is None
    assert prefer_output([_dev(3, "bluez_output", outputs=2)], "") is None
