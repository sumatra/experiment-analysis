from sumatra import OptimizeClient
import pendulum


def extract_start_time(experiment_id):
    if experiment_id:
        parts = experiment_id.split("-")
        if len(parts) == 2:
            return pendulum.from_format(parts[1], "YYYYMMDDHHmmss")


def get_start_time(client):
    thirty_days_ago = pendulum.now().subtract(days=30)
    start_times = []
    for opt in client.list_optimizations():
        for exp in client.list_experiences(opt["id"]):
            start_times.append(extract_start_time(exp["experimentId"]))
    first_time = min(t for t in start_times if t)
    if first_time:
        return max(thirty_days_ago, first_time)


def get_conversions(client):
    start_time = get_start_time(client)
    if not start_time:
        return None
    start_date = start_time.format("YYYY/MM/DD")
    sql = f"""
    with pages as (select event_id
                    , CAST(from_iso8601_timestamp(event_ts) as timestamp)                event_ts
                    , try_cast(json_extract_scalar(features, '$.session_id') as varchar) session_id
                    , try_cast(json_extract_scalar(features, '$.user_id') as varchar)    user_id
                    , try_cast(json_extract(features, '$.audiences') as array(varchar))  audiences
                    , assignment['optimization']                                         optimization
                    , assignment['experience']                                           experience
                    , assignment['variant']                                              variant
                    , assignment['experiment']                                           experiment
               from (event_log
                   cross join unnest(try_cast(json_extract(features, '$.optimizations') as array(map(varchar, varchar)))) t (assignment))
               where (event_type = 'page') and event_date >= '{start_date}')
   , goals as (select event_id
                     , cast(from_iso8601_timestamp(event_ts) as timestamp)                  event_ts
                     , try_cast(json_extract_scalar(features, '$.session_id') as varchar)   session_id
                     , try_cast(json_extract_scalar(features, '$.user_id') as varchar)      user_id
                     , try_cast(json_extract_scalar(features, '$.optimization') as varchar) optimization
                from event_log
                where (event_type = 'goal' or event_type = 'click') and event_date >= '{start_date}')
    select pages.optimization            optimization,
        pages.experience              experience,
        pages.experiment              experiment,
        pages.variant                 variant,
        count(distinct pages.user_id) user_count,
        count(distinct goals.user_id) converted_user_count
    from (pages
        left join goals on ((pages.user_id = goals.user_id) and (pages.optimization = goals.optimization) and (goals.event_ts >= pages.event_ts)))
    group by 1, 2, 3, 4
    """
    return client.query_athena(sql)


if __name__ == "__main__":
    client = OptimizeClient("console.sumatra.ai", workspace="timothy")
    print(get_conversions(client))
