--
-- PostgreSQL database dump
--

\restrict ADcUnLQRgW8M8axWhUGyn1NQYq1zHhoCfPBmr4ad24ZWDhvRiapKhqFvzr9k3ap

-- Dumped from database version 15.18
-- Dumped by pg_dump version 15.18

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: decision_logs; Type: TABLE; Schema: public; Owner: poker
--

CREATE TABLE public.decision_logs (
    id integer NOT NULL,
    created_at timestamp without time zone,
    round_idx integer,
    hole_cards json,
    community_cards json,
    bet_history json,
    pot_size integer,
    stack_size integer,
    action_taken character varying,
    strategy_used character varying
);


ALTER TABLE public.decision_logs OWNER TO poker;

--
-- Name: decision_logs_id_seq; Type: SEQUENCE; Schema: public; Owner: poker
--

CREATE SEQUENCE public.decision_logs_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.decision_logs_id_seq OWNER TO poker;

--
-- Name: decision_logs_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: poker
--

ALTER SEQUENCE public.decision_logs_id_seq OWNED BY public.decision_logs.id;


--
-- Name: decision_logs id; Type: DEFAULT; Schema: public; Owner: poker
--

ALTER TABLE ONLY public.decision_logs ALTER COLUMN id SET DEFAULT nextval('public.decision_logs_id_seq'::regclass);


--
-- Data for Name: decision_logs; Type: TABLE DATA; Schema: public; Owner: poker
--

COPY public.decision_logs (id, created_at, round_idx, hole_cards, community_cards, bet_history, pot_size, stack_size, action_taken, strategy_used) FROM stdin;
1	2026-05-15 09:13:46.052049	1	["Ah", "Kh"]	["Qh", "Jh", "Th", "2c", "3d"]	[20]	100	200	raise	mccfr_blueprint
2	2026-05-15 09:15:22.044562	1	["Ah", "Kh"]	["Qh", "Jh", "Th", "2c", "3d"]	[10]	50	200	raise	mccfr_blueprint
3	2026-05-15 09:15:22.060235	2	["Ah", "Kh"]	["Qh", "Jh", "Th", "2c", "3d"]	[20]	100	200	raise	mccfr_blueprint
4	2026-05-15 09:15:22.065545	3	["Ah", "Kh"]	["Qh", "Jh", "Th", "2c", "3d"]	[30]	150	200	raise	mccfr_blueprint
5	2026-05-15 09:15:22.075314	4	["Ah", "Kh"]	["Qh", "Jh", "Th", "2c", "3d"]	[40]	200	200	raise	mccfr_blueprint
6	2026-05-15 09:15:22.084175	5	["Ah", "Kh"]	["Qh", "Jh", "Th", "2c", "3d"]	[50]	250	200	raise	mccfr_blueprint
7	2026-05-15 09:15:22.094206	6	["Ah", "Kh"]	["Qh", "Jh", "Th", "2c", "3d"]	[60]	300	200	raise	mccfr_blueprint
8	2026-05-15 09:15:22.10635	7	["Ah", "Kh"]	["Qh", "Jh", "Th", "2c", "3d"]	[70]	350	200	raise	mccfr_blueprint
9	2026-05-15 09:15:22.115275	8	["Ah", "Kh"]	["Qh", "Jh", "Th", "2c", "3d"]	[80]	400	200	raise	mccfr_blueprint
10	2026-05-15 09:15:22.126381	9	["Ah", "Kh"]	["Qh", "Jh", "Th", "2c", "3d"]	[90]	450	200	raise	mccfr_blueprint
11	2026-05-15 09:15:22.136033	10	["Ah", "Kh"]	["Qh", "Jh", "Th", "2c", "3d"]	[100]	500	200	raise	mccfr_blueprint
\.


--
-- Name: decision_logs_id_seq; Type: SEQUENCE SET; Schema: public; Owner: poker
--

SELECT pg_catalog.setval('public.decision_logs_id_seq', 11, true);


--
-- Name: decision_logs decision_logs_pkey; Type: CONSTRAINT; Schema: public; Owner: poker
--

ALTER TABLE ONLY public.decision_logs
    ADD CONSTRAINT decision_logs_pkey PRIMARY KEY (id);


--
-- Name: ix_decision_logs_id; Type: INDEX; Schema: public; Owner: poker
--

CREATE INDEX ix_decision_logs_id ON public.decision_logs USING btree (id);


--
-- PostgreSQL database dump complete
--

\unrestrict ADcUnLQRgW8M8axWhUGyn1NQYq1zHhoCfPBmr4ad24ZWDhvRiapKhqFvzr9k3ap

