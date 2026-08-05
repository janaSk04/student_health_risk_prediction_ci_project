import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface StudentHealthInput {
  sleep_duration: number | null;
  heart_rate: number | null;
  bmi: number | null;
  calorie_expenditure: number | null;
  step_count: number | null;
  exercise_duration: number | null;
  water_intake: number | null;
  diet_type: string | null;
  stress_level: string | null;
  sleep_quality: string | null;
  physical_activity_level: string | null;
  smoking_alcohol: string | null;
  gender: string | null;
}

export interface PredictionResponse {
  health_condition: string;
  probabilities: Record<string, number>;
  message: string;
}

@Injectable({ providedIn: 'root' })
export class PredictionService {
  // FastAPI backend URL
  private readonly apiUrl = 'http://127.0.0.1:8000';

  constructor(private http: HttpClient) {}

  predict(payload: StudentHealthInput): Observable<PredictionResponse> {
    return this.http.post<PredictionResponse>(`${this.apiUrl}/predict`, payload);
  }
}
