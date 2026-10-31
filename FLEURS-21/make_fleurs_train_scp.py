from pathlib import Path
from typing import Dict, List, Tuple


# ============================================================
# 1. 路径设置
# ============================================================

DATA_ROOT = Path(
    "/train4/ztj/Language_Identification/"
    "FleursMini-wespeaker/data/2_16k16bit1channel"
)

OUTPUT_DIR = Path(
    "/train4/ztj/Language_Identification/"
    "Fleurs21-Balanced/data/4_train_test"
)

# 输出的 train 集 wav.scp
TRAIN_SCP_PATH = OUTPUT_DIR / "train.scp"

# 统计信息
TRAIN_STATS_PATH = OUTPUT_DIR / "train_stats.txt"


# ============================================================
# 2. 固定选择21个语种
# ============================================================

SELECTED_LANGUAGES = [
    # Western Europe
    "en_us",         # English
    "es_419",        # Spanish
    "fi_fi",         # Finnish

    # Eastern Europe
    "ru_ru",         # Russian
    "ka_ge",         # Georgian
    "hy_am",         # Armenian

    # Central Asia / Middle East / North Africa
    "ar_eg",         # Arabic
    "fa_ir",         # Persian
    "kk_kz",         # Kazakh

    # Sub-Saharan Africa
    "sw_ke",         # Swahili
    "ha_ng",         # Hausa
    "yo_ng",         # Yoruba

    # South Asia
    "hi_in",         # Hindi
    "bn_in",         # Bengali
    "ta_in",         # Tamil

    # South-East Asia
    "id_id",         # Indonesian
    "vi_vn",         # Vietnamese
    "th_th",         # Thai

    # Chinese / Japanese / Korean
    "cmn_hans_cn",   # Mandarin Chinese
    "ja_jp",         # Japanese
    "ko_kr",         # Korean
]


# ============================================================
# 3. 生成 utterance ID
# ============================================================

def get_utterance_id(
    language: str,
    wav_path: Path,
) -> str:
    """
    生成 Kaldi/WeSpeaker 格式的 utterance ID。

    格式：
        {lang}_{basename}

    示例：
        language = ja_jp
        wav_path = 10632268744167710712.wav

    返回：
        ja_jp_10632268744167710712
    """
    return f"{language}_{wav_path.stem}"


# ============================================================
# 4. 检查21个指定语种目录
# ============================================================

def find_selected_language_dirs() -> List[Path]:
    """
    检查指定的21个语种目录是否全部存在。
    """

    language_dirs: List[Path] = []
    missing_languages: List[str] = []

    for language in SELECTED_LANGUAGES:
        language_dir = DATA_ROOT / language

        if language_dir.is_dir():
            language_dirs.append(language_dir)
        else:
            missing_languages.append(language)

    if missing_languages:
        raise RuntimeError(
            "\n以下指定语种没有在 DATA_ROOT 中找到：\n"
            + "\n".join(
                f"  - {language}"
                for language in missing_languages
            )
        )

    if len(language_dirs) != 21:
        raise RuntimeError(
            f"期望21个语种，实际找到{len(language_dirs)}个。"
        )

    return language_dirs


# ============================================================
# 5. 主程序
# ============================================================

def main() -> None:

    if not DATA_ROOT.exists():
        raise FileNotFoundError(
            f"数据根目录不存在：{DATA_ROOT}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # 获取固定21个语种目录
    # --------------------------------------------------------

    language_dirs = find_selected_language_dirs()

    print("=" * 80)
    print("FLEURS-21 Train SCP 提取")
    print("=" * 80)

    print(f"DATA_ROOT：{DATA_ROOT}")
    print(f"OUTPUT_DIR：{OUTPUT_DIR}")
    print(f"选择语种数量：{len(language_dirs)}")

    print()
    print("选择的21个语种：")

    for index, language_dir in enumerate(
        language_dirs,
        start=1,
    ):
        print(
            f"{index:>2}. "
            f"{language_dir.name}"
        )

    print()
    print("=" * 80)

    # --------------------------------------------------------
    # 保存全部 train.scp 条目
    # --------------------------------------------------------

    all_train_entries: List[
        Tuple[str, Path]
    ] = []

    # 每个语种统计信息
    language_statistics: List[
        Tuple[str, int]
    ] = []

    # 检查 utterance ID 是否重复
    utterance_id_to_path: Dict[
        str,
        Path,
    ] = {}

    # ========================================================
    # 6. 遍历21个语种
    # ========================================================

    for language_dir in language_dirs:

        language = language_dir.name

        train_dir = (
            language_dir
            / "train"
        )

        if not train_dir.exists():
            raise FileNotFoundError(
                f"{language} 缺少 train 文件夹："
                f"{train_dir}"
            )

        # ----------------------------------------------------
        # 查找当前语种所有 train WAV
        # ----------------------------------------------------

        wav_files = sorted(
            wav_path
            for wav_path in train_dir.rglob("*")
            if wav_path.is_file()
            and wav_path.suffix.lower() == ".wav"
        )

        if not wav_files:
            raise RuntimeError(
                f"{language} 的 train 文件夹中"
                f"没有 WAV 文件：{train_dir}"
            )

        # ----------------------------------------------------
        # 加入 train.scp
        # ----------------------------------------------------

        for wav_path in wav_files:

            utterance_id = get_utterance_id(
                language=language,
                wav_path=wav_path,
            )

            absolute_path = (
                wav_path.resolve()
            )

            # 检查第一列 utterance ID 是否重复
            if (
                utterance_id
                in utterance_id_to_path
            ):
                previous_path = (
                    utterance_id_to_path[
                        utterance_id
                    ]
                )

                raise RuntimeError(
                    "发现重复 utterance ID。\n"
                    f"utterance ID：{utterance_id}\n"
                    f"文件1：{previous_path}\n"
                    f"文件2：{absolute_path}"
                )

            utterance_id_to_path[
                utterance_id
            ] = absolute_path

            all_train_entries.append(
                (
                    utterance_id,
                    absolute_path,
                )
            )

        # ----------------------------------------------------
        # 保存语种统计信息
        # ----------------------------------------------------

        language_statistics.append(
            (
                language,
                len(wav_files),
            )
        )

        print(
            f"{language:<16}"
            f"train WAV 数量："
            f"{len(wav_files):>5}"
        )

    # ========================================================
    # 7. 按 utterance ID 排序
    # ========================================================

    all_train_entries.sort(
        key=lambda item: item[0]
    )

    # ========================================================
    # 8. 写入 train.scp
    # ========================================================

    with TRAIN_SCP_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:

        for (
            utterance_id,
            wav_path,
        ) in all_train_entries:

            file.write(
                f"{utterance_id} "
                f"{wav_path}\n"
            )

    # ========================================================
    # 9. 写入统计信息
    # ========================================================

    total_train_count = (
        len(all_train_entries)
    )

    with TRAIN_STATS_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "Language\t"
            "TrainWavCount\n"
        )

        file.write(
            "=" * 40
            + "\n"
        )

        for (
            language,
            count,
        ) in language_statistics:

            file.write(
                f"{language}\t"
                f"{count}\n"
            )

        file.write(
            "=" * 40
            + "\n"
        )

        file.write(
            f"Number of languages\t"
            f"{len(language_statistics)}\n"
        )

        file.write(
            f"Total train WAV count\t"
            f"{total_train_count}\n"
        )

    # ========================================================
    # 10. 最终输出
    # ========================================================

    print()
    print("=" * 80)
    print("FLEURS-21 train.scp 生成完成")
    print("=" * 80)

    print(
        f"成功处理语种数量："
        f"{len(language_statistics)}"
    )

    print(
        f"Train WAV 音频总数："
        f"{total_train_count}"
    )

    print()
    print(
        f"train.scp："
        f"{TRAIN_SCP_PATH}"
    )

    print(
        f"统计文件："
        f"{TRAIN_STATS_PATH}"
    )


if __name__ == "__main__":
    main()