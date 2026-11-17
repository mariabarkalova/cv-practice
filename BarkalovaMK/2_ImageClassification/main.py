import logging
from collections.abc import Sequence

from cv_practice.cli import cli_argument_parser

LOGGER = logging.getLogger("cv_practice") #созд журнал

#приним аргум из ком стр
def main(argv: Sequence[str] | None = None) -> int:
    args = cli_argument_parser(argv)
    LOGGER.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    #для сообщ напри INFO Прочитано 100/1333 обучающих изображений
    if not LOGGER.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
        LOGGER.addHandler(handler)
    LOGGER.propagate = False

    try:
        args.handler(args) #запуск run_train, run_test или run_predict (из pipelines.py), передавая им все настройки
    #обраб ошибок
    except KeyboardInterrupt:
        LOGGER.warning("Работа прервана.")
        return 130
    except Exception as error:
        LOGGER.error("%s", error, exc_info=args.verbose)
        return 1
    return 0


if __name__ == "__main__":
    exit_code = main()
    if exit_code:
        raise SystemExit(exit_code)
