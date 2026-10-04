def create_representation(features, tree):

    return {
        "schema_version": "1.0",
        "language": "python",
        "features": features,
        "tree": tree
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