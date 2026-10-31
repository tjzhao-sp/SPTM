import hashlib
import random
import wave
from pathlib import Path
from typing import Dict, List, Tuple


# ============================================================
# 1. 路径及参数设置
# ============================================================

DATA_ROOT = Path(
    "/train4/ztj/Language_Identification/"
    "FleursMini-wespeaker/data/2_16k16bit1channel"
)

OUTPUT_DIR = Path(
    "/train4/ztj/Language_Identification/"
    "Fleurs21-Balanced/data/3_enroll_dev_split"
)

ENROLL_SCP_PATH = OUTPUT_DIR / "enroll.scp"
DEV_SCP_PATH = OUTPUT_DIR / "dev.scp"
STATS_PATH = OUTPUT_DIR / "split_stats.txt"


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


# 每个语种 enrollment 目标时长
TARGET_ENROLL_SECONDS = 300.0

# 固定随机种子
RANDOM_SEED = 2026

# 至少给dev保留的音频数量
MIN_DEV_FILES = 1


# ============================================================
# 3. 基础函数
# ============================================================

def get_wav_duration(wav_path: Path) -> float:
    """读取WAV文件时长，单位为秒。"""
    try:
        with wave.open(str(wav_path), "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            num_frames = wav_file.getnframes()

            if sample_rate <= 0:
                raise ValueError(f"非法采样率：{sample_rate}")

            return num_frames / float(sample_rate)

    except (wave.Error, EOFError, OSError, ValueError) as error:
        raise RuntimeError(
            f"无法读取 WAV 文件：{wav_path}\n"
            f"错误信息：{error}"
        ) from error


def get_language_seed(language: str) -> int:
    """
    根据全局随机种子和语种名称生成稳定随机种子。
    保证不同语种独立随机，同时每次运行结果一致。
    """
    seed_string = f"{RANDOM_SEED}_{language}"
    digest = hashlib.sha256(seed_string.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def get_utterance_id(language: str, wav_path: Path) -> str:
    """
    生成utterance ID。

    格式：
        {language}_{wav_basename}

    例如：
        ja_jp_123456789
    """
    return f"{language}_{wav_path.stem}"


# ============================================================
# 4. 检查21个语种目录
# ============================================================

def find_selected_language_dirs() -> List[Path]:
    """检查21个指定语种是否全部存在。"""

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
# 5. Enrollment / Dev 划分
# ============================================================

def split_enrollment_and_dev(
    wav_infos: List[Tuple[Path, float]],
    language: str,
) -> Tuple[
    List[Tuple[Path, float]],
    List[Tuple[Path, float]],
]:
    """
    将一个语种的dev划分为：
        enrollment：累计时长尽量接近300秒
        dev：其余全部音频

    流程：
        1. 固定种子随机打乱
        2. 顺序加入enrollment
        3. 到达300秒附近时比较加入前后哪个更接近300秒
        4. 其余全部进入dev
    """

    if len(wav_infos) <= MIN_DEV_FILES:
        raise RuntimeError(
            f"{language} 只有 {len(wav_infos)} 条音频，"
            "无法同时划分 enrollment 和 dev。"
        )

    shuffled_infos = wav_infos.copy()

    random_generator = random.Random(
        get_language_seed(language)
    )

    random_generator.shuffle(
        shuffled_infos
    )

    enroll_infos: List[Tuple[Path, float]] = []
    enroll_duration = 0.0

    for wav_path, duration in shuffled_infos:

        remaining_count = (
            len(wav_infos)
            - len(enroll_infos)
            - 1
        )

        if remaining_count < MIN_DEV_FILES:
            break

        duration_after_add = (
            enroll_duration
            + duration
        )

        # 还没有达到目标时长
        if duration_after_add < TARGET_ENROLL_SECONDS:
            enroll_infos.append(
                (
                    wav_path,
                    duration,
                )
            )

            enroll_duration = (
                duration_after_add
            )

            continue

        # 加入前距离目标值
        difference_before = abs(
            TARGET_ENROLL_SECONDS
            - enroll_duration
        )

        # 加入后距离目标值
        difference_after = abs(
            TARGET_ENROLL_SECONDS
            - duration_after_add
        )

        # 选择更加接近300秒的情况
        if (
            not enroll_infos
            or difference_after <= difference_before
        ):
            enroll_infos.append(
                (
                    wav_path,
                    duration,
                )
            )

            enroll_duration = (
                duration_after_add
            )

        break

    if not enroll_infos:
        raise RuntimeError(
            f"{language} 未成功抽取 enrollment 音频。"
        )

    enroll_path_set = {
        wav_path
        for wav_path, _ in enroll_infos
    }

    dev_infos = [
        (
            wav_path,
            duration,
        )
        for wav_path, duration in wav_infos
        if wav_path not in enroll_path_set
    ]

    if len(dev_infos) < MIN_DEV_FILES:
        raise RuntimeError(
            f"{language} 划分后剩余 dev 音频数量不足。"
        )

    return (
        enroll_infos,
        dev_infos,
    )


# ============================================================
# 6. 主程序
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

    language_dirs = (
        find_selected_language_dirs()
    )

    print("=" * 90)
    print("FLEURS-21 Enrollment / Dev Split")
    print("=" * 90)

    print(
        f"DATA_ROOT：{DATA_ROOT}"
    )

    print(
        f"OUTPUT_DIR：{OUTPUT_DIR}"
    )

    print(
        f"选择语种数量："
        f"{len(language_dirs)}"
    )

    print(
        f"每个语种 enrollment 目标时长："
        f"{TARGET_ENROLL_SECONDS:.2f} 秒"
    )

    print(
        f"随机种子："
        f"{RANDOM_SEED}"
    )

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
    print("=" * 90)

    all_enroll_entries: List[
        Tuple[str, Path]
    ] = []

    all_dev_entries: List[
        Tuple[str, Path]
    ] = []

    statistics: List[Dict] = []

    utterance_id_to_path: Dict[
        str,
        Path,
    ] = {}

    # ========================================================
    # 7. 遍历21个语种
    # ========================================================

    for language_dir in language_dirs:

        language = (
            language_dir.name
        )

        dev_dir = (
            language_dir
            / "dev"
        )

        if not dev_dir.exists():
            raise FileNotFoundError(
                f"{language} 缺少 dev 文件夹："
                f"{dev_dir}"
            )

        wav_files = sorted(
            wav_path
            for wav_path in dev_dir.rglob("*")
            if wav_path.is_file()
            and wav_path.suffix.lower() == ".wav"
        )

        if not wav_files:
            raise RuntimeError(
                f"{language} 的 dev 文件夹中"
                f"没有 WAV 文件：{dev_dir}"
            )

        # ----------------------------------------------------
        # 读取音频时长
        # ----------------------------------------------------

        wav_infos: List[
            Tuple[Path, float]
        ] = []

        for wav_path in wav_files:

            duration = (
                get_wav_duration(
                    wav_path
                )
            )

            wav_infos.append(
                (
                    wav_path,
                    duration,
                )
            )

        # ----------------------------------------------------
        # 划分 Enrollment / Dev
        # ----------------------------------------------------

        (
            enroll_infos,
            dev_infos,
        ) = split_enrollment_and_dev(
            wav_infos=wav_infos,
            language=language,
        )

        enroll_duration = sum(
            duration
            for _, duration
            in enroll_infos
        )

        dev_duration = sum(
            duration
            for _, duration
            in dev_infos
        )

        total_duration = (
            enroll_duration
            + dev_duration
        )

        # ----------------------------------------------------
        # 排序
        # ----------------------------------------------------

        enroll_infos.sort(
            key=lambda item:
            get_utterance_id(
                language,
                item[0],
            )
        )

        dev_infos.sort(
            key=lambda item:
            get_utterance_id(
                language,
                item[0],
            )
        )

        # ----------------------------------------------------
        # Enrollment
        # ----------------------------------------------------

        for wav_path, _ in enroll_infos:

            utterance_id = (
                get_utterance_id(
                    language,
                    wav_path,
                )
            )

            absolute_path = (
                wav_path.resolve()
            )

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

            all_enroll_entries.append(
                (
                    utterance_id,
                    absolute_path,
                )
            )

        # ----------------------------------------------------
        # Dev
        # ----------------------------------------------------

        for wav_path, _ in dev_infos:

            utterance_id = (
                get_utterance_id(
                    language,
                    wav_path,
                )
            )

            absolute_path = (
                wav_path.resolve()
            )

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

            all_dev_entries.append(
                (
                    utterance_id,
                    absolute_path,
                )
            )

        # ----------------------------------------------------
        # 保存统计
        # ----------------------------------------------------

        statistics.append(
            {
                "language": language,
                "total_count": len(
                    wav_infos
                ),
                "total_duration": (
                    total_duration
                ),
                "enroll_count": len(
                    enroll_infos
                ),
                "enroll_duration": (
                    enroll_duration
                ),
                "dev_count": len(
                    dev_infos
                ),
                "dev_duration": (
                    dev_duration
                ),
            }
        )

        print(
            f"{language:<16}"
            f"总数={len(wav_infos):>4}条，"
            f"enroll="
            f"{len(enroll_infos):>3}条/"
            f"{enroll_duration:>7.2f}秒，"
            f"dev="
            f"{len(dev_infos):>4}条/"
            f"{dev_duration:>8.2f}秒"
        )

    # ========================================================
    # 8. 全局排序
    # ========================================================

    all_enroll_entries.sort(
        key=lambda item: item[0]
    )

    all_dev_entries.sort(
        key=lambda item: item[0]
    )

    # ========================================================
    # 9. 检查 Enrollment / Dev 重叠
    # ========================================================

    enroll_ids = {
        utterance_id
        for utterance_id, _
        in all_enroll_entries
    }

    dev_ids = {
        utterance_id
        for utterance_id, _
        in all_dev_entries
    }

    overlap_ids = (
        enroll_ids
        & dev_ids
    )

    if overlap_ids:
        raise RuntimeError(
            f"Enrollment 与 Dev 存在 "
            f"{len(overlap_ids)} 条重复音频。"
        )

    # ========================================================
    # 10. 写入 enroll.scp
    # ========================================================

    with ENROLL_SCP_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:

        for (
            utterance_id,
            wav_path,
        ) in all_enroll_entries:

            file.write(
                f"{utterance_id} "
                f"{wav_path}\n"
            )

    # ========================================================
    # 11. 写入 dev.scp
    # ========================================================

    with DEV_SCP_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:

        for (
            utterance_id,
            wav_path,
        ) in all_dev_entries:

            file.write(
                f"{utterance_id} "
                f"{wav_path}\n"
            )

    # ========================================================
    # 12. 总体统计
    # ========================================================

    total_enroll_count = (
        len(all_enroll_entries)
    )

    total_dev_count = (
        len(all_dev_entries)
    )

    total_enroll_duration = sum(
        item[
            "enroll_duration"
        ]
        for item in statistics
    )

    total_dev_duration = sum(
        item[
            "dev_duration"
        ]
        for item in statistics
    )

    # ========================================================
    # 13. 写入统计文件
    # ========================================================

    with STATS_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "Language\t"
            "TotalCount\t"
            "TotalDurationSeconds\t"
            "EnrollCount\t"
            "EnrollDurationSeconds\t"
            "DevCount\t"
            "DevDurationSeconds\n"
        )

        for item in statistics:

            file.write(
                f"{item['language']}\t"
                f"{item['total_count']}\t"
                f"{item['total_duration']:.2f}\t"
                f"{item['enroll_count']}\t"
                f"{item['enroll_duration']:.2f}\t"
                f"{item['dev_count']}\t"
                f"{item['dev_duration']:.2f}\n"
            )

        file.write(
            "=" * 110
            + "\n"
        )

        file.write(
            f"Number of languages\t"
            f"{len(statistics)}\n"
        )

        file.write(
            f"Target enrollment duration per language\t"
            f"{TARGET_ENROLL_SECONDS:.2f}\n"
        )

        file.write(
            f"Random seed\t"
            f"{RANDOM_SEED}\n"
        )

        file.write(
            f"Total enrollment files\t"
            f"{total_enroll_count}\n"
        )

        file.write(
            f"Total enrollment duration seconds\t"
            f"{total_enroll_duration:.2f}\n"
        )

        file.write(
            f"Total dev files\t"
            f"{total_dev_count}\n"
        )

        file.write(
            f"Total dev duration seconds\t"
            f"{total_dev_duration:.2f}\n"
        )

        file.write(
            f"Enroll/Dev overlap\t"
            f"{len(overlap_ids)}\n"
        )

    # ========================================================
    # 14. 最终输出
    # ========================================================

    print()
    print("=" * 90)
    print("FLEURS-21 划分完成")
    print("=" * 90)

    print(
        f"成功处理语种数量："
        f"{len(statistics)}"
    )

    print(
        f"Enrollment 音频数量："
        f"{total_enroll_count}"
    )

    print(
        f"Enrollment 总时长："
        f"{total_enroll_duration:.2f} 秒"
    )

    print(
        f"Enrollment 平均每语种时长："
        f"{total_enroll_duration / len(statistics):.2f} 秒"
    )

    print(
        f"Dev 音频数量："
        f"{total_dev_count}"
    )

    print(
        f"Dev 总时长："
        f"{total_dev_duration:.2f} 秒"
    )

    print(
        f"Enrollment / Dev 重叠："
        f"{len(overlap_ids)}"
    )

    print()
    print(
        f"enroll.scp："
        f"{ENROLL_SCP_PATH}"
    )

    print(
        f"dev.scp："
        f"{DEV_SCP_PATH}"
    )

    print(
        f"统计文件："
        f"{STATS_PATH}"
    )


if __name__ == "__main__":
    main()