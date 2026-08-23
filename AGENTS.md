# Working agreement

- Keep this project independent of external local projects and datasets.
- Never commit model weights, training datasets, secrets, or `.env` files.
- Preserve the preprocessing contract: any change to `backend/app/config.py` must be checked against the model's training pipeline.
- Run `pytest` and `ruff check backend frontend tests` after Python changes.
- Keep API responses backwards compatible unless a versioned API change is intentional.
