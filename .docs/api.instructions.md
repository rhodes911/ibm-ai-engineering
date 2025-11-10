# API Guidelines

- Use Flask or FastAPI for building REST APIs.
- Separate routing, business logic, and data access layers.
- Validate and sanitize all input; handle exceptions gracefully.
- Document each endpoint with its path, parameters, and example responses.
- Use environment variables for secrets, API keys, and other sensitive configuration; never hard-code credentials.
- Return clear JSON responses with appropriate HTTP status codes.
- Write unit tests for each endpoint using pytest.
