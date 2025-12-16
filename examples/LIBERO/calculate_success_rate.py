import argparse
import re
from collections import defaultdict


def parse_log(logfile_path):
    """
    Parses the log file to calculate success rates for each task.
    """
    # Regex to find the task name and the success status
    # We use re.search() so it can find the pattern anywhere in the line
    task_regex = re.compile(r"Task: (.*)")
    success_regex = re.compile(r"Success: (True|False)")

    # Use defaultdict to easily initialize new tasks
    # Structure: {'task_name': {'success': 0, 'total': 0}}
    task_stats = defaultdict(lambda: {"success": 0, "total": 0})

    current_task = None

    try:
        with open(logfile_path, "r") as f:
            for line in f:
                # Check if a new task is defined
                task_match = task_regex.search(line)
                if task_match:
                    # .strip() removes any leading/trailing whitespace from the task name
                    current_task = task_match.group(1).strip()
                    continue  # Go to the next line after identifying a task

                # Check for a success status
                success_match = success_regex.search(line)

                # We only log a success/failure if we know what task it belongs to
                if success_match and current_task:
                    status_str = success_match.group(1)

                    # Increment total attempts for this task
                    task_stats[current_task]["total"] += 1

                    # Increment success count if True
                    if status_str == "True":
                        task_stats[current_task]["success"] += 1

    except FileNotFoundError:
        print(f"Error: Log file not found at '{logfile_path}'")
        return
    except Exception as e:
        print(f"An error occurred while parsing the file: {e}")
        return

    # --- Print the final report ---
    print("--- Task success rate report ---")

    if not task_stats:
        print("No tasks or success records were found in the logs.")
        return

    total_episodes_all_tasks = 0
    total_success_all_tasks = 0

    for task, stats in task_stats.items():
        total = stats["total"]
        success = stats["success"]

        total_episodes_all_tasks += total
        total_success_all_tasks += success

        if total > 0:
            rate = (success / total) * 100
            print(f"\ntask: {task}")
            print(f"success rate: {rate:.2f}% ({success} / {total})")
        else:
            print(f"\ntask: {task}")
            print("success rate: N/A (0 try)")

    # Print overall summary
    print("\n" + "=" * 30)
    if total_episodes_all_tasks > 0:
        overall_rate = (total_success_all_tasks / total_episodes_all_tasks) * 100
        print(
            f"Overall success rate (all tasks): {overall_rate:.2f}% ({total_success_all_tasks} / {total_episodes_all_tasks})"
        )
    else:
        print("No attempts were logged.")
    print("=" * 30)


if __name__ == "__main__":
    # Set up argument parsing to accept the log file path
    parser = argparse.ArgumentParser(description="从日志文件中解析每个任务的成功率。")
    parser.add_argument("logfile", type=str, help="指向 .log 文件的路径")

    args = parser.parse_args()

    parse_log(args.logfile)
