import numpy as np
import time
import os

times = np.load(os.path.join(os.path.dirname(__file__), 'inference', 'times.npz'), allow_pickle=True)
descs = np.load(os.path.join(os.path.dirname(__file__), 'inference', 'descs.npz'), allow_pickle=True)

def compute_map_features(ref_map_images):
    ref_desc = []
    for img in ref_map_images:
        t = times[img]
        desc = descs[img]
        print('Encode Time: ', t)
        ref_desc.append(desc)
        print(desc.shape)
    return ref_desc

def compute_query_desc(image_query):
    query_desc = descs[image_query]
    print(query_desc.shape)
    return query_desc
       
def perform_VPR(query_desc,ref_map_features):
    all_scores=[]
    for i in range(len(ref_map_features)):
        t1=time.time()    
        query_desc=query_desc.astype('float64')
        ref_desc=ref_map_features[i].astype('float64')
        match_score=np.dot(query_desc,ref_desc.T)
        t2=time.time()
        print('Custom NPZ tm:',t2-t1)
        all_scores.append(match_score)
    
    return np.amax(all_scores), np.argmax(all_scores),  np.asarray(all_scores).reshape(len(ref_map_features))
        
    