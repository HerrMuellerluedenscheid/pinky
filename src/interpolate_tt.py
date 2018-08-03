from pyrocko import util
import numpy as num
import scipy
from pyrocko.model.event import load_events
from util import append_to_dict
from pyrocko import orthodrome

from pyrocko.gui.marker import PhaseMarker, EventMarker, associate_phases_to_events


def get_reference_location(events):
    n_events = len(events)
    lats = num.empty(n_events)
    lons = num.empty(n_events)
    depths = num.empty(n_events)

    for i_ev, ev in enumerate(events):
        lats[i_ev] = ev.lat
        lons[i_ev] = ev.lon
        depths[i_ev] = ev.depth


    return(num.min(lats), num.max(lons), num.max(depths))

if __name__ == '__main__': 
    fn_events = 'events.pf'
    fn_markers = 'hypodd_markers.pf'
    markers = PhaseMarker.load_markers(fn_markers)
    events = load_events(fn_events)
    event_markers = [EventMarker(e) for e in events]
    markers.extend(event_markers)
    associate_phases_to_events(markers)
    stat_nsl_list = ['.NKC.']
    wanted_phase = 'P'


    min_lat, min_lon, max_depth = get_reference_location(events)


    
    by_event = {}
    by_event_noTT = {}

    for m in markers:
        if not isinstance(m, PhaseMarker):
            continue
        if not m.get_phasename().upper() == wanted_phase:
            continue
        event = m.get_event()
        if not event:
            continue

        append_to_dict(by_event, event, m)

    by_station = {}

    coordinates = {}
    for event, markers in by_event.items():
        x,y = orthodrome.latlon_to_ne(min_lat, min_lon,
                                    event.lat, event.lon)


        for m in markers:
                nsl = m.one_nslc()[:3]
                tt_p = m.tmin - event.time
        
                append_to_dict(by_station, nsl, (x,y,event.depth, tt_p))

    #print(by_station)

    # dict containing the not not-picked events for all stations 
    '''
    by_event_nott = {}
    for m in markers:
        if isinstance(m, PhaseMarker):
            if m.get_phasename().upper() == wanted_phase:
                continue
    '''


    # iterate over station, use interpolation 
    interp_by_station = {}
    for st, coor_t_list in by_station.items():
        xyz_array = num.empty((len(coor_t_list),3))
        xyz_array[:,0] = [tup[0] for tup in coor_t_list]
        xyz_array[:,1] = [tup[1] for tup in coor_t_list]
        xyz_array[:,2] = [tup[2] for tup in coor_t_list]

        tt = [tup[3] for tup in coor_t_list]

        interp_by_station[st] = scipy.interpolate.LinearNDInterpolator(
            points=xyz_array,
            values=tt,
            )

    new_tt_by_st = {}
    for st, interp in interp_by_station.items():
        for ev in events:
            if ev in by_event.keys():
                if st in [m.one_nslc()[0:3] for m in by_event[ev]]:
                    continue

            x,y = orthodrome.latlon_to_ne(min_lat, min_lon,
                                          ev.lat, ev.lon)

            interp_ = interp(x, y, ev.depth)
            print(interp_)
            if interp == num.nan:
                print('nan - ',x,y,ev.depth)
            else:
                append_to_dict(new_tt_by_st, st, (ev, interp_))

    #print(new_tt_by_st)



# picks anschauen! sinnvoll?
    