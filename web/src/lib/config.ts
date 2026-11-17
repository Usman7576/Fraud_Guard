// Server-side settings. Never import this from a client component.
import "server-only";

export const API_URL = (process.env.FRAUDGUARD_API_URL ?? "http://127.0.0.1:8000").replace(/\/$/, "");
export const API_PREFIX = "/api/v1";
