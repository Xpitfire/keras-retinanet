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
            suggestions = []
            for s_obj_key, s_obj_value in r_obj.suggestions.items():
                suggestions.append(s_obj_key)
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
