# Dashboards

## Refresh behaviour

Dashboards created from uploaded files refresh when the underlying file is
replaced. Dashboards connected to a database use the schedule configured under
Dashboard Settings. Available schedules are manual, hourly, daily, and weekly.

## Stale dashboard data

If a dashboard shows old data, first check its last successful refresh time.
Then confirm that the data source is available and that the refresh schedule is
enabled. Error E310 means a scheduled refresh failed. The refresh history page
contains the failure time and a short diagnostic message.

## Manual refresh

Users with Editor or Administrator access can select Refresh Now. Viewers
cannot start a refresh. If a manual refresh also fails, record the error code
and contact support.

