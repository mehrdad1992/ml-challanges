
-- Section1

select D.family_name, D.name, D.times_won from (select family_name,given_name as name, count(*) as times_won, B.player_id from awards as A join award_winners as B on A.award_id=B.award_id join players as C on B.player_id=C.player_id where award_name="Golden Boot" group by family_name, given_name, B.player_id order by times_won desc, B.player_id asc) as D;

-- Section2

select C.team_name, sum(score) as total_score from (select case when B.team_name='West Germany' then 'Germany' else B.team_name end as team_name, -A.position+5 as score from tournament_standings as A left join teams as B on A.team_id=B.team_id) as C group by C.team_name order by total_score desc, team_name asc limit 10;

-- Section3

select B.tournament_id as tournament_id, B.year as tournament_year, red_card_count from (select tournament_id, year from tournaments) as B join (select A.tournament_id, sum(red_card) as red_card_count from (select tournament_id, red_card from bookings where red_card=1) as A group by tournament_id) as C on B.tournament_id=C.tournament_id order by red_card_count desc, B.tournament_id asc;

-- Section4

   your fourth query here

-- Section5

   your fifth query here

-- Section6

   your sixth query here
