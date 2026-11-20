"""Per-photo undo/redo snapshots, persisted with the reviewed frame."""
def history_state(frame):
    if 'review_history' in frame:return frame['review_history'],frame['review_history_index']
    current=sorted(frame.get('excluded_object_ids',[]))
    states=[[],current] if current else [[]]
    return states,len(states)-1

def history_edit(frame,excluded):
    states,index=history_state(frame);value=sorted(set(excluded))
    states=[list(v) for v in states[:index+1]]
    if states[-1]!=value:states.append(value)
    return dict(review_history=states,review_history_index=len(states)-1)

def history_move(frame,direction):
    states,index=history_state(frame);target=index+direction
    if direction not in (-1,1) or not 0<=target<len(states):return None
    return dict(review_history=[list(v) for v in states],review_history_index=target),list(states[target])
