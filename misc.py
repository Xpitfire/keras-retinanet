from functools import wraps
from flask import request, current_app

def jsonify(classification_results):
    # parse the json response
    # TODO: replace with API - currently solved like this because no proper model objects 
    # defined and response is working process
    json = '{ "results": [ '
    for i, result in enumerate(classification_results):
        json += '{ '
        # time
        json += '"time": '
        json += '"'
        json += str(result.time_elapsed)
        json += '", '
        # url
        json += '"url": '
        json += '"'
        json += result.url
        json += '", '
        # captions
        json += '"captions": [ '
        for j, item in enumerate(result.caption_list):
            id, label, score, box = item
            json += '{ '
            # id
            json += '"id": '
            json += '"'
            json += str(id)
            json += '", '
            # label
            json += '"label": '
            json += '"'
            json += label
            json += '", '
            # score
            json += '"score": '
            json += '"'
            json += str(score)
            json += '", '
            # box
            json += '"top-left": '
            json += '"'
            json += str(box[0]) + ';' + str(box[1]) # (x1, y1)
            json += '", '
            json += '"bottom-right": '
            json += '"'
            json += str(box[2]) + ';' + str(box[3]) # (x2, y2)
            json += '"'
            json += ' }'
            if j < len(result.caption_list)-1:
                json += ', '
        json += ' ] '
        json += ' }'
        if i < len(classification_results)-1:
            json += ', '
    json += ' ] }'
    return json

def jsonp(func):
    # TODO: fix jsonp workaround - currently used to avoid cors issue 
    """Wraps JSONified output for JSONP requests."""
    @wraps(func)
    def decorated_function(*args, **kwargs):
        callback = request.args.get('callback', False)
        if callback:
            # TODO: fix workaround for wrong python string generation
            # python adds a b prefix to any string although UTF-8 encoding specified
            # this removes the prefix and corrects the response
            content = '{}({});'.format(callback, str(func(*args, **kwargs).data)).replace("(b", "(")
            mimetype = 'application/javascript'
            return current_app.response_class(content, mimetype=mimetype)
        else:
            return func(*args, **kwargs)
    return decorated_function