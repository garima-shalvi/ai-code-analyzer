from agents.reporting_agent import format_report


def generate_report(result):
    
    if "status" not in result:
        raise ValueError("Pass the dict returned by orchestrator.analyze_source().")
    print(format_report(result))