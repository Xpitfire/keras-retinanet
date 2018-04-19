# Define Asset DTO class
class QueryAsset:
    def __init__(self, asset_id, asset_url, asset_encoding=None, similarity_index=None):
        self.asset_id = asset_id
        self.asset_url = asset_url
        self.asset_encoding = asset_encoding
        self.similarity_index = similarity_index

    def is_complete(self):
        return self.asset_id and self.asset_url and self.similarity_index is not None and (self.asset_encoding.size if \
                                                                                           self.asset_encoding is not None else None) is not None

    def __str__(self):
        output = "Asset: "
        output += "\n--id: " + str(self.asset_id) if self.asset_id is not None else "None"
        output += "\n--url: "
        output += self.asset_url if self.asset_url is not None else "None"
        output += "\n--sim. index: "
        output += str(self.similarity_index) if self.similarity_index is not None else "None"
        output += "\n--encoding shape: "
        output += str(self.asset_encoding.shape) if self.asset_encoding is not None else "None"
        return output
