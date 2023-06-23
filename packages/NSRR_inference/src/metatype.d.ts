import * as index from "wonder-ts-meta-typing/src/number/index"

// type ConvertToNumber<Value extends any> = Value extends number ? Value : never
type ConvertToNumber<Value extends any> = Value extends number ? Value : number

export type Mult<N1 extends number, N2 extends number> = ConvertToNumber<index.Mult<N1, N2>>