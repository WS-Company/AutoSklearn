from argparse import ArgumentParser

import numpy as np
import pandas as pd

from autolearn import DataPreprocessor


if __name__ == "__main__":
    parser = ArgumentParser(add_help=True)
    parser.add_argument(
        "filename",
        metavar="FILENAME",
        type=str,
        help="Имя исходного CSV-файла с данными"
    )
    parser.add_argument(
        "--output-filename", "-o",
        metavar="FILENAME",
        type=str,
        default=None,
        help="Имя файла для сохранения преобразованных данных"
    )
    parser.add_argument(
        "--pca",
        metavar="N_COMPONENTS",
        type=int,
        default=None,
        help="Привести данные к заданному числу главных компонент"
    )
    parser.add_argument(
        "--auto-normalize-first",
        action="store_const",
        const=True,
        default=False,
        help="Выполнить автонормализацию перед другими преобразованиями"
    )
    parser.add_argument(
        "--auto-normalize-last",
        action="store_const",
        const=True,
        default=False,
        help="Выполнить автонормализацию после других преобразований"
    )
    parser.add_argument(
        "--verbosity",
        metavar="VALUE",
        type=int,
        default=0,
        help="Количество выводимой на экран отладочной информации от 0 до 9"
    )
    args = parser.parse_args()
    proc = DataPreprocessor(
        pca = args.pca,
        auto_normalize_before_transform=args.auto_normalize_first,
        auto_normalize_after_transform=args.auto_normalize_last
    )
    (X, y) = proc.read_data(args.filename)
    proc.fit(X, y)
    X = proc.transform(X, y)
    output = pd.DataFrame(
        np.hstack([X, y.reshape(-1, 1)])
    )
    if args.output_filename:
        output.to_csv(args.output_filename, index=False, header=False)
    print(output)
