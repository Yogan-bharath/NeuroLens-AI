export interface APIResponse<T> {
  data: T | null;
  error: APIError | null;
  status: number;
}

export interface APIError {
  code: string;
  message: string;
  details?: Record<string, unknown>;
}
