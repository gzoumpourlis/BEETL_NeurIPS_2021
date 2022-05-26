import mne

def add_bipolar(epochs):
    # https://mne.tools/stable/generated/mne.set_bipolar_reference.html

    epochs = mne.set_bipolar_reference(epochs,
                                       anode=['C3'],
                                       cathode=['C4'],
                                       ch_name=['C3Bi'],
                                       ch_info=None,
                                       drop_refs=False,
                                       copy=False,
                                       verbose=None
                                       )
    epochs = mne.set_bipolar_reference(epochs,
                                       anode=['C4'],
                                       cathode=['C3'],
                                       ch_name=['C4Bi'],
                                       ch_info=None,
                                       drop_refs=False,
                                       copy=False,
                                       verbose=None
                                       )

    return epochs
