from ai_agent.models.runtime import ExecutionResult, ExecutionOutcome, ExecutionStatus

r = ExecutionResult.from_error('test error')
print('status:', r.status)
print('success property:', r.success)
print('is_success:', r.is_success)
print('error:', r.error)
print('outcome:', r.outcome)
print('should_continue:', r.should_continue)

print()
r2 = ExecutionResult.success('ok', ExecutionOutcome.STOP)
print('success result:')
print('  status:', r2.status)
print('  success property:', r2.success)
print('  is_success:', r2.is_success)
