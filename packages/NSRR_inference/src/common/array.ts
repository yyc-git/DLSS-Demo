export let range = (a: number, b: number) => {
    let result = []

    for (let i = a; i <= b; i++) {
        result.push(i)
    }

    return result
}