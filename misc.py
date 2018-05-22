import functools
from flask import request, current_app

from models import Content


def jsonp(func):
    # TODO: fix jsonp workaround - currently used to avoid cors issue
    """Wraps JSONified output for JSONP requests."""

    @functools.wraps(func)
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


def classify_get_req_to_content(asset_id, url):
    return Content({
        "assets": [
            {
                "asset-id": asset_id,
                "url": url
            }
        ]
    })

