/** What the front door answers: its routes' types (src/gen/routes.ts, made from glue/serve/src/home/routes.rs), and
 *  the page's own views of them. */

export type { Available, Config, Maintenance, ModelState, Proc, Service, SettingRow } from "./gen/routes.js";
import type { CardState, ModelView, State as StateAnswer, DriveView, DrivesView, Spec as SpecAnswer } from "./gen/routes.js";

import type { Clocks } from "./ages.js";
/** Optional clock fields allow the page to read older servers too. */
export interface Instance extends Clocks { gpu?: number | null; gpus?: number[] | null; port?: number | null; busy: boolean }
export type Card = Omit<CardState, "vouched"> & { vouched: (CardState["vouched"][number] & Clocks)[] };
export type Model = ModelView & Clocks & { instances?: Instance[] };
export type Drive = Omit<DriveView, "measured"> & { measured?: { at: number; bytes: number; seconds: number; file: string; mb_s: number } | null };
export type State = Omit<StateAnswer, "cards" | "models" | "drives"> & { cards: Card[]; models: Model[]; drives: Omit<DrivesView, "drives" | "move"> & { drives: Drive[]; move?: Move | null } };


/** An operation of the OpenAPI document, as the API tab reads it */
export interface Operation {
  tags: string[];
  operationId?: string;
  summary: string;
  requestBody?: { content: { "application/json": { example?: unknown; schema?: { properties?: Record<string, { type: string; description?: string; enum?: string[] }>; required?: string[] } } } };
  parameters?: unknown[];
  "x-curl"?: string;
}

/** The API tab's answer, its document read as the tab reads it */
export type Spec = Omit<SpecAnswer, "spec"> & {
  spec: { info: { description: string; version: string }; servers: { url: string }[]; paths: Record<string, Record<string, Operation>> };
};

/** What saving the settings answers (an action's answer: its own fields) */
export interface ConfigSaved { changed: string[]; restart: string[]; note?: string | null }

export interface Move {
  model: string;
  to: string;
  from: string[];
  state: "copying" | "registry" | "removing" | "done" | "failed" | "cancelled";
  running: boolean;
  files: number;
  file_index: number;
  file: string | null;
  bytes: number;
  done: number;
  work: number;
  pct: number;
  rate_mb_s: number;
  eta_s: number | null;
  started: number;
  seconds: number;
  error: string | null;
  warning: string | null;
}
