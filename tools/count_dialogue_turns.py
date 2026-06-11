import os
import pandas as pd
from collections import defaultdict

import warnings

warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')

class_file_path = "../data/300class/origin"
score_file_path = "../data/300class/score.xlsx"


def read_file_by_code(file_code, file_path_root):
    """根据编号读取对应文件"""
    for filename in os.listdir(file_path_root):
        if filename.startswith(f"{file_code}_"):
            file_path = os.path.join(file_path_root, filename)
            if filename.endswith(".xlsx"):
                return pd.read_excel(file_path)
            elif filename.endswith(".csv"):
                return pd.read_csv(file_path)
    return None


def get_lesson_info(lesson_id, score_file_path):
    """从评分文件中获取课程信息（学科、级别、阶段、年级等）"""
    try:
        score_list = pd.read_excel(score_file_path)
        result = score_list[score_list['课例id'] == int(lesson_id)]
        if not result.empty:
            lesson_data = result.iloc[0].to_dict( )
            return {
                '学科': lesson_data.get('学科', '未知'),
                '级别': lesson_data.get('级别', '未知'),
                '阶段': lesson_data.get('阶段', '未知'),
                '年级': lesson_data.get('年级', '未知')
            }
    except Exception as e:
        print(f"读取课程信息错误: {e}")
    return {'学科': '未知', '级别': '未知', '阶段': '未知', '年级': '未知'}


def count_conversation_turns(data):
    """统计话轮、教师话轮和学生话轮"""
    if data is None or data.empty:
        return 0, 0, 0

    total_turns = len(data)
    teacher_turns = 0
    student_turns = 0

    for _, row in data.iterrows( ):
        role = str(row['角色']).strip( )
        if '教师' in role or '老师' in role or 'teacher' in role.lower( ):
            teacher_turns += 1
        elif '学生' in role or 'student' in role.lower( ):
            student_turns += 1

    return total_turns, teacher_turns, student_turns


def generate_statistics(class_file_path, score_file_path):
    """生成统计信息"""
    stats = {
        '按学科统计': defaultdict(lambda: {'课程数': 0, '总话轮': 0, '教师话轮': 0, '学生话轮': 0}),
        '按级别统计': defaultdict(lambda: {'课程数': 0, '总话轮': 0, '教师话轮': 0, '学生话轮': 0}),
        '按阶段统计': defaultdict(lambda: {'课程数': 0, '总话轮': 0, '教师话轮': 0, '学生话轮': 0}),
        '按年级统计': defaultdict(lambda: {'课程数': 0, '总话轮': 0, '教师话轮': 0, '学生话轮': 0}),
        '总体统计': {'课程数': 0, '总话轮': 0, '教师话轮': 0, '学生话轮': 0}
    }

    # 获取所有课程文件
    all_files = [f for f in os.listdir(class_file_path) if f.endswith(('.xlsx', '.csv'))]
    course_codes = sorted(list(set(int(f.split('_')[0]) for f in all_files)))

    print(f"发现 {len(course_codes)} 个课程文件")

    # 统计每个课程
    for course_code in course_codes:
        data = read_file_by_code(course_code, class_file_path)
        if data is not None:
            total_turns, teacher_turns, student_turns = count_conversation_turns(data)
            lesson_info = get_lesson_info(course_code, score_file_path)

            subject = lesson_info['学科']
            level = lesson_info['级别']
            stage = lesson_info['阶段']
            grade = lesson_info['年级']

            # 按学科统计
            stats['按学科统计'][subject]['课程数'] += 1
            stats['按学科统计'][subject]['总话轮'] += total_turns
            stats['按学科统计'][subject]['教师话轮'] += teacher_turns
            stats['按学科统计'][subject]['学生话轮'] += student_turns

            # 按级别统计
            stats['按级别统计'][level]['课程数'] += 1
            stats['按级别统计'][level]['总话轮'] += total_turns
            stats['按级别统计'][level]['教师话轮'] += teacher_turns
            stats['按级别统计'][level]['学生话轮'] += student_turns

            # 按阶段统计
            stats['按阶段统计'][stage]['课程数'] += 1
            stats['按阶段统计'][stage]['总话轮'] += total_turns
            stats['按阶段统计'][stage]['教师话轮'] += teacher_turns
            stats['按阶段统计'][stage]['学生话轮'] += student_turns

            # 按年级统计
            stats['按年级统计'][grade]['课程数'] += 1
            stats['按年级统计'][grade]['总话轮'] += total_turns
            stats['按年级统计'][grade]['教师话轮'] += teacher_turns
            stats['按年级统计'][grade]['学生话轮'] += student_turns

            # 总体统计
            stats['总体统计']['课程数'] += 1
            stats['总体统计']['总话轮'] += total_turns
            stats['总体统计']['教师话轮'] += teacher_turns
            stats['总体统计']['学生话轮'] += student_turns

    return stats


def print_statistics(stats):
    """打印统计信息"""
    print("\n" + "=" * 80)
    print("课堂对话统计结果")
    print("=" * 80)

    # 总体统计
    print("\n1. 总体统计:")
    print("-" * 60)
    overall = stats['总体统计']
    teacher_ratio = overall['教师话轮'] / overall['总话轮'] if overall['总话轮'] > 0 else 0
    student_ratio = overall['学生话轮'] / overall['总话轮'] if overall['总话轮'] > 0 else 0

    print(f"总课程数: {overall['课程数']}")
    print(f"总话轮数: {overall['总话轮']}")
    print(f"教师话轮: {overall['教师话轮']} ({teacher_ratio:.1%})")
    print(f"学生话轮: {overall['学生话轮']} ({student_ratio:.1%})")
    print(f"师生话轮比: {teacher_ratio:.1%} : {student_ratio:.1%}")

    # 按学科统计
    print("\n2. 按学科统计:")
    print("-" * 70)
    print(f"{'学科':<10} {'课程数':<8} {'总话轮':<8} {'教师话轮':<10} {'学生话轮':<10} {'师生比':<12}")
    print("-" * 70)

    for subject, subject_stats in sorted(stats['按学科统计'].items( )):
        total = subject_stats['总话轮']
        teacher_ratio = subject_stats['教师话轮'] / total if total > 0 else 0
        student_ratio = subject_stats['学生话轮'] / total if total > 0 else 0
        ratio_str = f"{teacher_ratio:.1%}:{student_ratio:.1%}"

        print(f"{subject:<10} {subject_stats['课程数']:<8} {total:<8} "
              f"{subject_stats['教师话轮']:<10} {subject_stats['学生话轮']:<10} {ratio_str:<12}")

    # 按级别统计
    print("\n3. 按级别统计:")
    print("-" * 70)
    print(f"{'级别':<10} {'课程数':<8} {'总话轮':<8} {'教师话轮':<10} {'学生话轮':<10} {'师生比':<12}")
    print("-" * 70)

    for level, level_stats in sorted(stats['按级别统计'].items( )):
        total = level_stats['总话轮']
        teacher_ratio = level_stats['教师话轮'] / total if total > 0 else 0
        student_ratio = level_stats['学生话轮'] / total if total > 0 else 0
        ratio_str = f"{teacher_ratio:.1%}:{student_ratio:.1%}"

        print(f"{level:<10} {level_stats['课程数']:<8} {total:<8} "
              f"{level_stats['教师话轮']:<10} {level_stats['学生话轮']:<10} {ratio_str:<12}")

    # 按阶段统计
    print("\n4. 按阶段统计:")
    print("-" * 70)
    print(f"{'阶段':<10} {'课程数':<8} {'总话轮':<8} {'教师话轮':<10} {'学生话轮':<10} {'师生比':<12}")
    print("-" * 70)

    for stage, stage_stats in sorted(stats['按阶段统计'].items( )):
        total = stage_stats['总话轮']
        teacher_ratio = stage_stats['教师话轮'] / total if total > 0 else 0
        student_ratio = stage_stats['学生话轮'] / total if total > 0 else 0
        ratio_str = f"{teacher_ratio:.1%}:{student_ratio:.1%}"

        print(f"{stage:<10} {stage_stats['课程数']:<8} {total:<8} "
              f"{stage_stats['教师话轮']:<10} {stage_stats['学生话轮']:<10} {ratio_str:<12}")

    # 按年级统计
    print("\n5. 按年级统计:")
    print("-" * 70)
    print(f"{'年级':<10} {'课程数':<8} {'总话轮':<8} {'教师话轮':<10} {'学生话轮':<10} {'师生比':<12}")
    print("-" * 70)

    for grade, grade_stats in sorted(stats['按年级统计'].items( )):
        total = grade_stats['总话轮']
        teacher_ratio = grade_stats['教师话轮'] / total if total > 0 else 0
        student_ratio = grade_stats['学生话轮'] / total if total > 0 else 0
        ratio_str = f"{teacher_ratio:.1%}:{student_ratio:.1%}"

        print(f"{grade:<10} {grade_stats['课程数']:<8} {total:<8} "
              f"{grade_stats['教师话轮']:<10} {grade_stats['学生话轮']:<10} {ratio_str:<12}")


if __name__ == "__main__":
    # 配置路径
    class_file_path = "../data/300class/origin"
    score_file_path = "../data/300class/score.xlsx"

    print("开始统计课堂对话数据...")

    # 生成统计信息
    stats = generate_statistics(class_file_path, score_file_path)

    # 打印统计结果
    print_statistics(stats)

    print("\n" + "=" * 80)
    print("统计完成！")
    print("=" * 80)