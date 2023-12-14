
SELECT
    ts.id AS test_set_id,
    too.id AS test_output_id,
    op.id AS operation_id,
    ts.user_id AS test_set_user_id,
    ts.content AS test_set_content,
    ts.created_at AS test_set_created_at,
    ts.updated_at AS test_set_updated_at,
    too.set_id AS test_output_set_id,
    too.user_id AS test_output_user_id,
    too.test_set_id AS test_output_test_set_id,
    too.operation_id AS test_output_operation_id,
    too.test_params AS test_output_test_params,
    too.test_result AS test_output_test_result,
    too.test_score AS test_output_test_score,
    too.test_metric_name AS test_output_test_metric_name,
    too.test_query AS test_output_test_query,
    too.test_output AS test_output_test_output,
    too.test_expected_output AS test_output_test_expected_output,
    too.test_context AS test_output_test_context,
    too.test_results AS test_output_test_results,
    too.created_at AS test_output_created_at,
    too.updated_at AS test_output_updated_at,
    op.user_id AS operation_user_id,
    op.operation_type AS operation_operation_type,
    op.operation_params AS operation_operation_params,
    op.test_set_id AS operation_test_set_id,
    op.created_at AS operation_created_at,
    op.updated_at AS operation_updated_at
FROM public.test_sets ts
JOIN public.test_outputs too ON ts.id = too.test_set_id
JOIN public.operations op ON op.id = too.operation_id
where operation_status ="COMPLETED";