/* USER CODE BEGIN Header */
/**
 ******************************************************************************
 * @file           : main.c
 * @brief          : Main program body
 ******************************************************************************
 * @attention
 *
 * Copyright (c) 2026 STMicroelectronics.
 * All rights reserved.
 *
 * This software is licensed under terms that can be found in the LICENSE file
 * in the root directory of this software component.
 * If no LICENSE file comes with this software, it is provided AS-IS.
 *
 ******************************************************************************
 */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "tim.h"
#include "usart.h"
#include "gpio.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include <string.h>
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */
typedef enum {
	DOOR_CLOSED_IDLE = 0, DOOR_OPENING, DOOR_HOLD_OPEN, DOOR_CLOSING
} DoorState_t;
/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
/* ===== 舵机参数（按你的机构实际调）===== */
#define ANGLE_MIN             0
#define ANGLE_MAX           180
#define OPEN_ANGLE           90
#define CLOSE_ANGLE           0

/* 你的设计里 -1 表示停止PWM释放扭矩（可选） */
#define SERVO_RELEASE_ANGLE  -1

/* ===== 时间参数（ms）===== */
#define TIMEOUT_MOVE_MS      500U    /* 开门/关门动作时间 */
#define HOLD_OPEN_MS        3000U    /* 开门保持时间 */

/* ===== 串口接收缓冲 ===== */
#define RX_BUFFER_SIZE        32U
/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */
/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/

/* USER CODE BEGIN PV */
volatile DoorState_t door_state = DOOR_CLOSED_IDLE;
volatile uint32_t state_timer = 0;

/* 串口单字节接收 + 行缓冲 */
volatile uint8_t rx_byte = 0;
char rx_buffer[RX_BUFFER_SIZE];
volatile uint8_t rx_index = 0;
/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
/* USER CODE BEGIN PFP */
void SystemClock_Config(void);

static void UART_SendStr(const char *s);
static void UART_SendLine(const char *s);
static void Process_CommandLine(const char *cmd);
void Servo_SetAngle(int16_t angle);
/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

/* USER CODE END 0 */

/**
 * @brief  The application entry point.
 * @retval int
 */
int main(void) {

	/* USER CODE BEGIN 1 */

	/* USER CODE END 1 */

	/* MCU Configuration--------------------------------------------------------*/

	/* Reset of all peripherals, Initializes the Flash interface and the Systick. */
	HAL_Init();

	/* USER CODE BEGIN Init */

	/* USER CODE END Init */

	/* Configure the system clock */
	SystemClock_Config();

	/* USER CODE BEGIN SysInit */

	/* USER CODE END SysInit */

	/* Initialize all configured peripherals */
	MX_GPIO_Init();
	MX_TIM3_Init();
	MX_USART1_UART_Init();
	/* USER CODE BEGIN 2 */
	/* 启动舵机PWM（按你的实际通道改） */
	HAL_TIM_PWM_Start(&htim3, TIM_CHANNEL_1);

	/* 上电默认关门，然后释放（可按你的机械结构调整） */
	Servo_SetAngle(CLOSE_ANGLE);
	HAL_Delay(300);
	Servo_SetAngle(SERVO_RELEASE_ANGLE);

	/* 清空接收缓冲 */
	memset(rx_buffer, 0, sizeof(rx_buffer));
	rx_index = 0;

	/* 启动串口单字节中断接收（关键） */
	HAL_UART_Receive_IT(&huart1, (uint8_t*) &rx_byte, 1);

	/* 上电提示（可选） */
	UART_SendLine("READY");
	/* USER CODE END 2 */

	/* Infinite loop */
	/* USER CODE BEGIN WHILE */
	while (1) {
		/* USER CODE END WHILE */
		/* USER CODE BEGIN 3 */
		uint32_t current_time = HAL_GetTick();

		switch (door_state) {
		case DOOR_CLOSED_IDLE:
			/* 空闲态：等待串口命令 */
			break;

		case DOOR_OPENING:
			if ((current_time - state_timer) >= TIMEOUT_MOVE_MS) {
				door_state = DOOR_HOLD_OPEN;
				state_timer = current_time;

				/* 动作完成回包（与树莓派door_brain.py对齐） */
				UART_SendLine("ACK:DONE");
			}
			break;

		case DOOR_HOLD_OPEN:
			if ((current_time - state_timer) >= HOLD_OPEN_MS) {
				door_state = DOOR_CLOSING;
				state_timer = current_time;

				/* 执行关门动作 */
				Servo_SetAngle(CLOSE_ANGLE);
			}
			break;

		case DOOR_CLOSING:
			if ((current_time - state_timer) >= TIMEOUT_MOVE_MS) {
				door_state = DOOR_CLOSED_IDLE;

				/* 释放舵机，降低发热/抖动（如你的机构需要持续力矩保持，改成 CLOSE_ANGLE） */
				Servo_SetAngle(SERVO_RELEASE_ANGLE);
			}
			break;

		default:
			/* 异常兜底 -> 关门并回空闲 */
			door_state = DOOR_CLOSED_IDLE;
			Servo_SetAngle(CLOSE_ANGLE);
			break;
		}

		/* 不要长延时，保持主循环可响应 */
		HAL_Delay(1);
	}
	/* USER CODE END 3 */
}

/**
 * @brief System Clock Configuration
 * @retval None
 */
void SystemClock_Config(void) {
	RCC_OscInitTypeDef RCC_OscInitStruct = { 0 };
	RCC_ClkInitTypeDef RCC_ClkInitStruct = { 0 };

	/** Initializes the RCC Oscillators according to the specified parameters
	 * in the RCC_OscInitTypeDef structure.
	 */
	RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSE;
	RCC_OscInitStruct.HSEState = RCC_HSE_ON;
	RCC_OscInitStruct.HSEPredivValue = RCC_HSE_PREDIV_DIV1;
	RCC_OscInitStruct.HSIState = RCC_HSI_ON;
	RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
	RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSE;
	RCC_OscInitStruct.PLL.PLLMUL = RCC_PLL_MUL9;
	if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK) {
		Error_Handler();
	}

	/** Initializes the CPU, AHB and APB buses clocks
	 */
	RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK | RCC_CLOCKTYPE_SYSCLK
			| RCC_CLOCKTYPE_PCLK1 | RCC_CLOCKTYPE_PCLK2;
	RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
	RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
	RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV2;
	RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;

	if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_2) != HAL_OK) {
		Error_Handler();
	}
}

/* USER CODE BEGIN 4 */

/**
 * @brief  串口发送字符串（阻塞式，简单可靠）
 */
static void UART_SendStr(const char *s) {
	if (s == NULL)
		return;
	HAL_UART_Transmit(&huart1, (uint8_t*) s, (uint16_t) strlen(s), 100);
}

/**
 * @brief  串口发送一行（自动加 \r\n）
 */
static void UART_SendLine(const char *s) {
	UART_SendStr(s);
	UART_SendStr("\r\n");
}

/**
 * @brief  处理完整的一行命令（由串口中断拼包后调用）
 *         支持:
 *           OPEN
 *           CMD:OPEN
 *           PING / CMD:PING
 */
static void Process_CommandLine(const char *cmd) {
	if ((strcmp(cmd, "OPEN") == 0) || (strcmp(cmd, "CMD:OPEN") == 0)) {

		if (door_state == DOOR_CLOSED_IDLE) {
			/* 进入开门状态 */
			door_state = DOOR_OPENING;
			state_timer = HAL_GetTick();

			/* 执行开门动作 */
			Servo_SetAngle(OPEN_ANGLE);

			/* 注意：不在这里立即ACK，等动作时间到再 ACK:DONE */
		}
		else {
			UART_SendLine("ERR:BUSY");
		}
		return;
	}

	if ((strcmp(cmd, "PING") == 0) || (strcmp(cmd, "CMD:PING") == 0)) {
		UART_SendLine("PONG");
		return;
	}

	UART_SendLine("ERR:CMD");
}

/**
 * @brief  舵机角度设置
 * @note   这里按常见 50Hz 舵机PWM 写法：
 *         - 周期20ms
 *         - 0.5ms~2.5ms 脉宽对应 0~180°
 *
 *         你原工程如果已有 Servo_SetAngle()，可直接替换为你的版本
 */
void Servo_SetAngle(int16_t angle) {
	/* 释放舵机（停止PWM输出脉宽） */
	if (angle < 0) {
		__HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_1, 0);
		return;
	}

	/* 限幅 */
	if (angle < ANGLE_MIN)
		angle = ANGLE_MIN;
	if (angle > ANGLE_MAX)
		angle = ANGLE_MAX;

	/*
	 * 假设 TIM3 已配置为：
	 *   PWM频率 50Hz (20ms)
	 *   ARR = 20000-1 （计数单位1us）
	 * 则:
	 *   0.5ms -> 500
	 *   2.5ms -> 2500
	 *
	 * 如果你的计数单位不是1us，请按实际定时器参数调整
	 */
	uint16_t pulse = (uint16_t) (500 + ((uint32_t) angle * 2000U) / 180U);
	__HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_1, pulse);
}

void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart) {
	if (huart->Instance == USART1) {

		if (rx_byte == '\r') {
			/* 忽略CR */
		} else if (rx_byte == '\n') {
			/* 一行结束，字符串终止 */
			rx_buffer[rx_index] = '\0';

			if (rx_index > 0) {
				Process_CommandLine(rx_buffer);
			}

			/* 清空索引，准备下一条命令 */
			rx_index = 0;
		} else {
			/* 入缓冲（防越界） */
			if (rx_index < (RX_BUFFER_SIZE - 1U)) {
				rx_buffer[rx_index++] = (char) rx_byte;
			} else {
				/* 溢出 -> 清空并报错 */
				rx_index = 0;
				UART_SendLine("ERR:CMD");
			}
		}

		/* 关键：继续接收下一字节 */
		HAL_UART_Receive_IT(&huart1, (uint8_t*) &rx_byte, 1);
	}
}

/**
 * @brief 串口错误回调（可选但推荐）
 * @note  发生错误后重启接收，提升稳定性
 */
void HAL_UART_ErrorCallback(UART_HandleTypeDef *huart) {
	if (huart->Instance == USART1) {
		HAL_UART_Receive_IT(&huart1, (uint8_t*) &rx_byte, 1);
	}
}

/* USER CODE END 4 */

/**
 * @brief  This function is executed in case of error occurrence.
 * @retval None
 */
void Error_Handler(void) {
	/* USER CODE BEGIN Error_Handler_Debug */
	/* User can add his own implementation to report the HAL error return state */
	__disable_irq();
	while (1) {
	}
	/* USER CODE END Error_Handler_Debug */
}

#ifdef USE_FULL_ASSERT
/**
 * @brief  Reports the name of the source file and the source line number
 *         where the assert_param error has occurred.
 * @param  file: pointer to the source file name
 * @param  line: assert_param error line source number
 * @retval None
 */
void assert_failed(uint8_t *file, uint32_t line)
{
	/* USER CODE BEGIN 6 */
	/* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
	/* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
