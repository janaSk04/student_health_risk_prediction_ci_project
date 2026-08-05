import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import {
  PredictionResponse,
  PredictionService,
  StudentHealthInput,
} from '../../services/prediction.service';

@Component({
  selector: 'app-predict',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './predict.component.html',
  styleUrl: './predict.component.css',
})
export class PredictComponent {
  // Form model bound to Angular template
  form: StudentHealthInput = {
    sleep_duration: 7,
    heart_rate: 72,
    bmi: 22.5,
    calorie_expenditure: 2200,
    step_count: 8000,
    exercise_duration: 40,
    water_intake: 2.2,
    diet_type: 'balanced',
    stress_level: 'medium',
    sleep_quality: 'average',
    physical_activity_level: 'moderate',
    smoking_alcohol: 'no',
    gender: 'female',
  };

  loading = false;
  errorMessage = '';
  result: PredictionResponse | null = null;

  constructor(private predictionService: PredictionService) {}

  onSubmit(): void {
    this.loading = true;
    this.errorMessage = '';
    this.result = null;

    this.predictionService.predict(this.form).subscribe({
      next: (response) => {
        this.result = response;
        this.loading = false;
      },
      error: (err) => {
        this.loading = false;
        this.errorMessage =
          err?.error?.detail ||
          err?.message ||
          'Prediction failed. Make sure the FastAPI server is running on port 8000.';
      },
    });
  }

  resultClass(): string {
    const label = this.result?.health_condition;
    if (label === 'fit') return 'ok';
    if (label === 'unhealthy') return 'danger';
    return 'warn';
  }
}
