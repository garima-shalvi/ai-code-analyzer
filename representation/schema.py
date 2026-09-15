def create_representation(features):
    return {
        "schema_version": "1.0",
        "language": "python",
        "features": features
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