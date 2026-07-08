"""
Alembic environment: reads the DB URL and target metadata from the app
itself (app.config.settings, app.db.models) rather than duplicating them in
alembic.ini, so migrations always run against the same DB the app/worker
will actually use.
"""

import os
import sys
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

# Make `app.*` importable when alembic is invoked from webapp/backend/
# (alembic.ini's prepend_sys_path=. covers this too, but this is explicit and
# doesn't depend on cwd matching alembic.ini's location).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from personal.basketball_analysis.webapp.backend.app.config import settings  # noqa: E402
from personal.basketball_analysis.webapp.backend.app.db.base import Base  # noqa: E402
from personal.basketball_analysis.webapp.backend.app.db import models  # noqa: E402,F401 -- import registers all models on Base.metadata

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

config.set_main_option("sqlalchemy.url", settings.database_url)


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
