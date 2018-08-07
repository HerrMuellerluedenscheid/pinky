from pyrocko.gui.marker import load_markers, save_markers, EventMarker
from pyrocko.gui.marker import associate_phases_to_events, PhaseMarker
import random

'''Split a marker file into a training and evaluation set.'''

fn_markers = 'markers_phases_events_nkc_P.pf'
split_rate_training = 0.75      # how much data to use for training

markers = load_markers(fn_markers)
associate_phases_to_events(markers)

event_markers = [m for m in markers if isinstance(m, EventMarker)]
markers = [m for m in markers if isinstance(m, PhaseMarker)]
random.shuffle(markers)

isplit = int(len(markers) * split_rate_training)
fn_markers = 'markers_phases_events_nkc_P.pf'
fn_out, suffix = fn_markers.rsplit('.', maxsplit=1)

fn_train = '%s_train.%s' % (fn_out, suffix)
fn_eval = '%s_eval.%s' % (fn_out, suffix)

# markers.extend(event_markers)
markers_train = markers[0: isplit]
markers_eval = markers[isplit:]
print('saved files:\n %s [%s markers], %s [%s markers]' % (
    fn_train, len(markers_train), fn_eval, len(markers_eval)))

markers_train.extend(event_markers)
markers_eval.extend(event_markers)

save_markers(markers_train, fn_train)
save_markers(markers_eval, fn_eval)

