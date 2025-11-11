# utils.py
import sys
import json
import re


def load_jsonl(file_path):
    """
    Loads JSONL data from file_path and returns a list of dicts.
    """
    try:
        with open(file_path, "r") as file:
            return [json.loads(line) for line in file]
    except Exception as e:
        print(f"Failed to load JSONL file: {e}")
        sys.exit(1)


def split_field(data, field_name):
    """
    Retrieve the specified field from the data dictionary and split it based on [split].
    Returns a list of statements.
    """
    field_value = data.get(field_name, "")
    if not field_value:
        return []
    if isinstance(field_value, str):
        # Use [split] as the delimiter with optional surrounding whitespace
        # sql_statements = [stmt.strip() for stmt in re.split(r'\[split\]\s*', field_value) if stmt.strip()]
        # return sql_statements
        return [field_value]
    elif isinstance(field_value, list):
        return field_value
    else:
        return []


def save_report_and_status(
    report_file_path,
    question_test_case_results,
    data_list,
    number_of_execution_errors,
    number_of_timeouts,
    number_of_assertion_errors,
    overall_accuracy,
    timestamp,
    big_logger,
):
    """
    Writes a detailed report to report_file_path and updates the 'status'/'error_message'
    fields in data_list based on question_test_case_results.
    """
    total_instances = len(data_list)
    try:
        with open(report_file_path, "w") as report_file:
            report_file.write("--------------------------------------------------\n")
            report_file.write(
                "BIRD CRITIC Stack Overflow Result Statistics (Postgres, Multi-Thread):\n"
            )
            report_file.write(f"Number of Instances: {total_instances}\n")
            report_file.write(
                f"Number of Execution Errors: {number_of_execution_errors}\n"
            )
            report_file.write(f"Number of Timeouts: {number_of_timeouts}\n")
            report_file.write(
                f"Number of Assertion Errors: {number_of_assertion_errors}\n"
            )
            total_errors = (
                number_of_execution_errors
                + number_of_timeouts
                + number_of_assertion_errors
            )
            report_file.write(f"Total Errors: {total_errors}\n")
            report_file.write(f"Overall Accuracy: {overall_accuracy:.2f}%\n")
            report_file.write(f"Timestamp: {timestamp}\n\n")

            # Iterate through each question result
            for i, q_res in enumerate(question_test_case_results):
                q_idx = q_res["instance_id"]
                t_total = q_res["total_test_cases"]
                t_pass = q_res["passed_test_cases"]
                t_fail = t_total - t_pass
                failed_list_str = (
                    ", ".join(q_res["failed_test_cases"]) if t_fail > 0 else "None"
                )

                eval_phase_note = ""
                if q_res.get("evaluation_phase_execution_error"):
                    eval_phase_note += " | Eval Phase: Execution Error"
                if q_res.get("evaluation_phase_timeout_error"):
                    eval_phase_note += " | Eval Phase: Timeout Error"
                if q_res.get("evaluation_phase_assertion_error"):
                    eval_phase_note += " | Eval Phase: Assertion Error"

                report_file.write(
                    f"Question_{q_idx}: ({t_pass}/{t_total}) test cases passed, "
                    f"failed test cases: {failed_list_str}{eval_phase_note}\n"
                )

                # Update data_list with pass/fail status
                if t_fail == 0:
                    # All test cases passed; no errors during evaluation
                    data_list[i]["status"] = "success"
                    data_list[i]["error_message"] = None
                else:
                    data_list[i]["status"] = "failed"
                    if failed_list_str != "None":
                        data_list[i]["error_message"] = f"{failed_list_str} failed"
                    else:
                        data_list[i]["error_message"] = eval_phase_note

            # Add statistical summary
            report_file.write("\n" + "=" * 50 + "\n")
            report_file.write("EXECUTION STATISTICS:\n")
            report_file.write("=" * 50 + "\n")

            # Collect executable instances (those without syntax or execution errors)
            executable_instances = []
            executable_with_assertion_error = []
            executable_success = []

            for q_res in question_test_case_results:
                instance_id = q_res["instance_id"]
                has_execution_error = q_res.get("evaluation_phase_execution_error", False)
                has_timeout_error = q_res.get("evaluation_phase_timeout_error", False)
                has_assertion_error = q_res.get("evaluation_phase_assertion_error", False)

                # Executable instances: no execution or timeout errors
                if not has_execution_error and not has_timeout_error:
                    executable_instances.append(instance_id)

                    if has_assertion_error:
                        executable_with_assertion_error.append(instance_id)
                    else:
                        executable_success.append(instance_id)

            # Write execution summary
            report_file.write(
                f"Total Executable Instances (no syntax/execution errors): {len(executable_instances)}\n"
            )
            report_file.write(
                f"Executable Instances with Assertion Errors: {len(executable_with_assertion_error)}\n"
            )
            report_file.write(
                f"Executable Instances Successfully Passed: {len(executable_success)}\n\n"
            )

            # List instance IDs
            if executable_instances:
                report_file.write("Executable Instance IDs:\n")
                for i, instance_id in enumerate(executable_instances, 1):
                    report_file.write(f"  {i}. {instance_id}")
                    if instance_id in executable_with_assertion_error:
                        report_file.write(" (Assertion Error)")
                    elif instance_id in executable_success:
                        report_file.write(" (Success)")
                    report_file.write("\n")
                report_file.write("\n")

            if executable_with_assertion_error:
                report_file.write("Executable Instances with Assertion Errors:\n")
                for i, instance_id in enumerate(executable_with_assertion_error, 1):
                    report_file.write(f"  {i}. {instance_id}\n")
                report_file.write("\n")

            if executable_success:
                report_file.write("Executable Instances Successfully Passed:\n")
                for i, instance_id in enumerate(executable_success, 1):
                    report_file.write(f"  {i}. {instance_id}\n")
                report_file.write("\n")
    except Exception as e:
        big_logger.error(f"Failed to write report: {e}")
