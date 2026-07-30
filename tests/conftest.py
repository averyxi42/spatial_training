"""Shared pytest options for the tests in this directory."""


def pytest_addoption(parser):
    parser.addoption(
        "--ckpt",
        action="store",
        default=None,
        help="checkpoint directory for tests that exercise a trained model "
             "(e.g. tests/test_vector_rollout.py's incremental-vs-single-shot parity)",
    )
