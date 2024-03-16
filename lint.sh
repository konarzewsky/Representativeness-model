set -xe

pyright .
flake8 .
black . --check
isort --profile black . --check-only
