from flask import jsonify


class ResponseEncoder(object):
    def __init__(self, obj):
        self.obj = obj

    def to_dict(self):
        res = {}
        result_list = []
        for r_obj in self.obj.result_list:
            result = {
                'asset-id': r_obj.asset_id,
                #'url': r_obj.url,
                'time': r_obj.time
            }
            captions = []
            for c_obj in r_obj.captions:
                captions.append({
                    'id': c_obj.caption_id,
                    'label': c_obj.label,
                    'score': c_obj.score,
                    'top-left': c_obj.top_left,
                    'bottom-right': c_obj.bottom_right
                })
            suggestions = {}
            for s_obj_key, s_obj_value in r_obj.suggestions.items():
                suggestions[s_obj_key] = {}
                #suggestions[s_obj_key]['url'] = s_obj_value.url
                frames = []
                for f_obj in s_obj_value.frames:
                    frames.append({
                        'frame-id': f_obj.frame_id,
                        #'faiss-idx': f_obj.faiss_idx,
                        #'url': f_obj.url
                    })
                if len(frames) > 0:
                    suggestions[s_obj_key]['frames'] = frames
            if len(captions) > 0:
                result['captions'] = captions
            if len(suggestions) > 0:
                result['similar-suggestions'] = suggestions
            result_list.append(result)
        res['results'] = result_list
        return res

    def to_json(self):
        dict_ = self.to_dict()
        return jsonify(dict_)
