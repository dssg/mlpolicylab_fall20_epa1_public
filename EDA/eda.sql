
-- the number of inspections by result
select found_violation_flag,
       count(*)                                                                            as "count",
       trunc(count(*) * 100.0 / (select count(*) from temp_tables.cmecomp3_ny_2014_16), 2) as "percentage"
from temp_tables.cmecomp3_ny_2014_16
group by found_violation_flag
;

-- violation type breakdown
select a.violation_type, b.violation_short_description, a.count
from (select violation_type, count(*) as "count"
      from temp_tables.cmecomp3_ny_2014_16
      where found_violation_flag = 'Y'
      group by violation_type) a
         join temp_tables.violation_info b
              on a.violation_type = b.violation_type
;

-- all NY facilities
select id_number from rcra.facilities where activity_location = 'NY';

-- NY facilities that weren't inspected between 2014-2016
select id_number, fed_waste_generator, transporter, active_site, operating_tsdf
from (select id_number, fed_waste_generator, transporter, active_site, operating_tsdf 
from rcra.facilities
where activity_location = 'NY') as a
where not exists
       (select handler_id
       from temp_tables.cmecomp3_ny_2014_16
where handler_id = id_number)
;

-- NY facilities that were inspected between 2014-2016
select id_number, fed_waste_generator, transporter, active_site, operating_tsdf
from (select id_number, fed_waste_generator, transporter, active_site, operating_tsdf 
from rcra.facilities
where activity_location = 'NY') as a
where exists
       (select handler_id
       from temp_tables.cmecomp3_ny_2014_16
where handler_id = id_number)
;

-- violation type distribution of facilities with violations during 2016
select cn.violation_type, COUNT(*)
from temp_tables.cmecomp3_ny_2016 cn 
where cn.found_violation_flag in ('Y')
group by cn.violation_type 
;

-- enforcement type distribution of facilities with violations during 2016
select cn.enforcement_type, COUNT(*)
from temp_tables.cmecomp3_ny_2016 cn 
where cn.found_violation_flag in ('Y')
group by cn.enforcement_type 
;

-- waste code group distribution of facilities with violations during 2016
select br.waste_code_group, COUNT(*)
from temp_tables.cmecomp3_ny_2016 cn, rcra.br_reporting br 
where br.state in ('NY') and cn.handler_id = br.handler_id and cn.found_violation_flag in ('Y')
group by br.waste_code_group 
;

-- management method distribution of facilities with violations during 2016
select br.management_method , COUNT(*)
from temp_tables.cmecomp3_ny_2016 cn, rcra.br_reporting br 
where br.state in ('NY') and cn.handler_id = br.handler_id and cn.found_violation_flag in ('Y')
group by br.management_method
;


-- base rate of active LQG in 2013
-- all active LQG in 2013
with all_handler as 
(
	select handler_id
	from nysdec_reports.si1 si1
	where report_year = '2013'
),
-- base rate of all active handlers in 2013, inspected in 2014 = 46.5%
formal as (select 
      c.handler_id,
      cast 
      (case 
      	when (count(enforcement_type in ('310', '210', '380', '385', '410', '420',
                                  '425', '430', '510', '520', '530', '610',
                                  '620', '630', '810', '820', '830', '840',
                                  '850', '860', '865')) = 0) then 'N'
        else 'Y' end as varchar)
      as has_formal_enforcement
from 
	all_handler as ah
	left join rcra.cmecomp3 c
    on ah.handler_id = c.handler_id
where 
    c.evaluation_start_date >= '2014-01-01' and c.evaluation_start_date <= '2014-12-31'
group by 
    c.handler_id)
select count(case when has_formal_enforcement = 'Y' then 1 end)::float / count(*) as BASE_RATE
from formal 
       
       
       
    
-- common sense model: rank by 1. days since last inspection 2. days since date_became_current in si2
-- all active LQG in 2013
with all_handler as 
(
	select handler_id
	from nysdec_reports.si1 si1
	where report_year = '2013'
	group by handler_id 
),
days as 
(
	select ah.handler_id, MIN('2014-01-01'::date - evaluation_start_date::date) as days_since_last_inspection
	from (all_handler as ah
		left join (select handler_id, evaluation_start_date from rcra.cmecomp3 c2
		where evaluation_start_date <= '2013-12-31') 
		as c
		on ah.handler_id = c.handler_id) 
	group by ah.handler_id
),
days_inspect as (select
	d.handler_id,
	cast (case 
		when d.days_since_last_inspection is null then 2147483647 else d.days_since_last_inspection
	end as bigint) as days_since_last_inspection_2
from days d
order by days_since_last_inspection_2 DESC),
days_current as (select ah.handler_id, MIN('2014-01-01'::date - s2.date_became_current::date) as days_since_became_current
from (all_handler as ah
	left join (select handler_id, date_became_current from nysdec_reports.si2 s
	where s.report_year = '2013') as s2
	on ah.handler_id = s2.handler_id)
group by ah.handler_id)
select di.handler_id, di.days_since_last_inspection_2 as days_since_inspection, dc.days_since_became_current
from days_inspect di left join days_current dc on di.handler_id = dc.handler_id
order by (di.days_since_last_inspection_2, dc.days_since_became_current) DESC
