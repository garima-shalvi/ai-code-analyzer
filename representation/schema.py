def create_representation(features, tree, code):
    return {
        "schema_version": "1.0",
        "language": "python",
        "features": features,
        "tree": tree,
        "code": code
    }

def create_error_representation(error_type, detail):

    return {
        "schema_version": "1.0",
        "language": "python",
        "error": {
            "type": error_type,
            "detail": detail
        }
    }